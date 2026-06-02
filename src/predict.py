"""
Module du doan (inference) mon an Viet Nam tu anh.

Module nay cung cap:
- FoodPredictor: Class chinh de du doan mon an tu anh
  + predict(): Du doan 1 anh -> ten mon + confidence + calories
  + predict_with_gradcam(): Du doan + heatmap giai thich
  + batch_predict(): Du doan nhieu anh
- quick_predict(): Ham tien ich 1 dong cho TV5 tich hop Gradio

Su dung:
    # Python:
    from src.predict import FoodPredictor
    predictor = FoodPredictor("checkpoints/best_model.pth")
    result = predictor.predict("anh_pho.jpg")

    # CLI:
    python src/predict.py --image anh_pho.jpg --checkpoint checkpoints/best_model.pth
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from src.model import create_model, GradCAM, get_target_layer
from src.preprocess import get_val_transform, IMAGENET_MEAN, IMAGENET_STD
from src.load_calories import load_full_nutrition
from src.food_metadata import get_display_name, get_story, get_ingredients


# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT: Path = Path(__file__).parent.parent
"""Thu muc goc cua project."""

DEFAULT_CHECKPOINT: Path = PROJECT_ROOT / "checkpoints" / "best_model.pth"
"""Duong dan mac dinh toi best checkpoint."""

SERVING_GRAMS: float = 350.0
"""Khoi luong mac dinh 1 phan an (gram) de tinh calories."""


# ============================================================================
# FOOD PREDICTOR
# ============================================================================


class FoodPredictor:
    """
    Predictor nhan dien mon an Viet Nam tu anh.

    Load model da train, transform anh, du doan class,
    va tra ve thong tin dinh duong + metadata mon an.

    Attributes:
        model: Model da load weights.
        transform: Val transform pipeline.
        class_names: Danh sach ten class (sorted).
        calories_data: Dict calories cua cac mon.
        device: torch.device.

    Example:
        >>> predictor = FoodPredictor("checkpoints/best_model.pth")
        >>> result = predictor.predict("test_pho.jpg")
        >>> print(f"{result['display_name']}: {result['confidence']:.1%}")
        Pho: 95.3%
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
    ) -> None:
        """
        Khoi tao predictor.

        Args:
            checkpoint_path: Duong dan toi model checkpoint (.pth).
            device: "auto" (tu dong), "cuda", hoac "cpu".

        Raises:
            FileNotFoundError: Neu checkpoint hoac calories file khong ton tai.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint khong ton tai: {checkpoint_path}")

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        print(f" Loading model tu {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Lay config tu checkpoint
        config = checkpoint.get("config", {})
        model_name = config.get("model_name", "efficientnet_b0")
        num_classes = config.get("num_classes", 29)

        # Tao model va load weights
        self.model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model_name = model_name
        self.num_classes = num_classes

        # Transform (giong val transform)
        self.transform = get_val_transform()

        # Class names - lay tu checkpoint hoac tu dataset folder
        self.class_names = self._get_class_names(checkpoint)

        # Nutrition data (macro + serving + micro-nutrients)
        try:
            self.calories_data = load_full_nutrition()
            print(f"[OK] Loaded full nutrition data: {len(self.calories_data)} classes (NIN Vietnam)")
        except Exception as e:
            print(f"[WARN] Khong load duoc nutrition data: {e}")
            self.calories_data = {}

        # Info
        best_acc = checkpoint.get("best_val_acc", "N/A")
        epoch = checkpoint.get("epoch", "N/A")
        print(f"[OK] Model loaded: {model_name}, {num_classes} classes")
        print(f"  Checkpoint epoch: {epoch}, best val acc: {best_acc}")
        print(f"  Device: {self.device}")

    def _get_class_names(self, checkpoint: dict) -> List[str]:
        """
        Lay danh sach class names.
        Uu tien: checkpoint > scan folder > hardcode.
        """
        # Thu lay tu checkpoint
        if "class_names" in checkpoint:
            return checkpoint["class_names"]

        # Scan tu data folder
        data_dir = PROJECT_ROOT / "data" / "raw" / "Images" / "Train"
        if data_dir.exists():
            names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            if len(names) == self.num_classes:
                return names

        # Fallback: tao tu FOOD_NAME_DISPLAY_MAP (bo Banh duc)
        from src.food_metadata import FOOD_NAME_DISPLAY_MAP
        names = sorted([k for k in FOOD_NAME_DISPLAY_MAP.keys() if k != "Banh duc"])
        return names[:self.num_classes]

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 3,
        serving_grams: float = SERVING_GRAMS,
    ) -> Dict:
        """
        Du doan mon an tu 1 anh.

        Args:
            image: Duong dan file anh, hoac PIL Image truc tiep.
            top_k: So luong top predictions tra ve.
            serving_grams: Khoi luong phan an (gram) de tinh calories.

        Returns:
            Dict chua:
            {
                "predicted_class": "Pho",          # Ten class (folder name)
                "display_name": "Pho",              # Ten tieng Viet co dau
                "confidence": 0.953,                # Xac suat du doan
                "top_k": [                          # Top K predictions
                    ("Pho", "Pho", 0.953),
                    ("Bun bo Hue", "Bun bo Hue", 0.031),
                    ...
                ],
                "nutrition": {                      # Dinh duong (per serving)
                    "food_name_vi": "Pho",
                    "kcal_per_100g": 135,
                    "serving_grams": 350,
                    "serving_kcal": 472.5,
                    "protein_g": 29.4,
                    "carb_g": 58.8,
                    "fat_g": 11.2,
                    "source": "USDA"
                },
                "story": "Dai su am thuc cua Viet Nam...",
                "ingredients": "Banh pho dep, thit bo..."
            }

        Example:
            >>> result = predictor.predict("anh_pho.jpg", top_k=5)
            >>> print(result["display_name"], result["confidence"])
        """
        # Load anh
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"image phai la str, Path, hoac PIL.Image, nhan: {type(image)}")

        # Transform
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)

        # Top-K
        top_probs, top_indices = probs.topk(min(top_k, self.num_classes), dim=1)
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

        # Xu ly truong hop batch_size=1 lam mat chieu
        if top_probs.ndim == 0:
            top_probs = np.array([top_probs.item()])
            top_indices = np.array([top_indices.item()])

        # Predicted class
        pred_idx = top_indices[0]
        pred_class = self.class_names[pred_idx]
        pred_confidence = float(top_probs[0])

        # Top-K list
        top_k_list = []
        for idx, prob in zip(top_indices, top_probs):
            class_name = self.class_names[idx]
            display = get_display_name(class_name)
            top_k_list.append((class_name, display, float(prob)))

        # Nutrition
        nutrition = self._get_nutrition(pred_class, serving_grams)

        # Metadata
        display_name = get_display_name(pred_class)
        story = get_story(pred_class)
        ingredients = get_ingredients(pred_class)

        return {
            "predicted_class": pred_class,
            "display_name": display_name,
            "confidence": pred_confidence,
            "top_k": top_k_list,
            "nutrition": nutrition,
            "story": story or "",
            "ingredients": ingredients or "",
        }

    def _get_nutrition(self, class_name: str, serving_grams: float) -> Dict:
        """
        Lay thong tin dinh duong day du cho 1 mon an.

        Bao gom: macro (kcal, protein, carb, fat) + vi chat (chat xo,
        cholesterol, canxi, sat, vitamin...) + thong tin suat an thuc te.
        Nguon: Vien Dinh duong Quoc gia Viet Nam (NIN).
        """
        empty_micros = {
            "fiber_g": 0, "cholesterol_mg": 0, "calcium_mg": 0,
            "phosphorus_mg": 0, "iron_mg": 0, "sodium_mg": 0,
            "potassium_mg": 0, "vitamin_a_mcg": 0,
            "vitamin_b1_mg": 0, "vitamin_c_mg": 0,
        }

        if class_name not in self.calories_data:
            return {
                "food_name_vi": get_display_name(class_name),
                "kcal_per_100g": 0,
                "serving_grams": serving_grams,
                "serving_kcal": 0,
                "protein_g": 0, "carb_g": 0, "fat_g": 0,
                "real_serving_weight_g": 0,
                "real_serving_kcal": 0,
                **empty_micros,
                "source": "N/A",
            }

        base = self.calories_data[class_name]
        scale = serving_grams / 100.0

        result = {
            "food_name_vi": base["food_name_vi"],
            "kcal_per_100g": base["kcal_per_100g"],
            "serving_grams": serving_grams,
            "serving_kcal": round(base["kcal_per_100g"] * scale, 1),
            # Macros (scaled theo serving_grams)
            "protein_g": round(base["protein_g"] * scale, 1),
            "carb_g": round(base["carb_g"] * scale, 1),
            "fat_g": round(base["fat_g"] * scale, 1),
            # Thong tin suat an thuc te (tu calories.csv)
            "real_serving_weight_g": base.get("serving_weight_g", 0),
            "real_serving_kcal": base.get("serving_kcal_total", 0),
            # Micro-nutrients (scaled theo serving_grams)
            "fiber_g": round(base.get("fiber_g", 0) * scale, 2),
            "cholesterol_mg": round(base.get("cholesterol_mg", 0) * scale, 2),
            "calcium_mg": round(base.get("calcium_mg", 0) * scale, 1),
            "phosphorus_mg": round(base.get("phosphorus_mg", 0) * scale, 1),
            "iron_mg": round(base.get("iron_mg", 0) * scale, 2),
            "sodium_mg": round(base.get("sodium_mg", 0) * scale, 1),
            "potassium_mg": round(base.get("potassium_mg", 0) * scale, 1),
            "vitamin_a_mcg": round(base.get("vitamin_a_mcg", 0) * scale, 2),
            "vitamin_b1_mg": round(base.get("vitamin_b1_mg", 0) * scale, 3),
            "vitamin_c_mg": round(base.get("vitamin_c_mg", 0) * scale, 2),
            "source": base["source"],
        }

        return result

    def predict_with_gradcam(
        self,
        image_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None,
        top_k: int = 3,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Du doan + tao Grad-CAM heatmap.

        Args:
            image_path: Duong dan toi file anh.
            save_path: Duong dan luu visualization. None = khong luu.
            top_k: So luong top predictions.

        Returns:
            Tuple[prediction_dict, heatmap_array]:
            - prediction_dict: Giong predict()
            - heatmap_array: np.ndarray (224, 224) gia tri [0, 1]

        Example:
            >>> result, heatmap = predictor.predict_with_gradcam("pho.jpg", save_path="gradcam_pho.png")
        """
        # Du doan truoc
        prediction = self.predict(image_path, top_k=top_k)

        # Tao Grad-CAM
        target_layer = get_target_layer(self.model, self.model_name)
        gradcam = GradCAM(self.model, target_layer)

        try:
            heatmap, _, _ = gradcam.visualize(
                image_path=image_path,
                transform=self.transform,
                class_names=self.class_names,
                save_path=save_path,
                show=False,
            )
        finally:
            gradcam.remove_hooks()

        return prediction, heatmap

    def batch_predict(
        self,
        image_paths: List[Union[str, Path]],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Du doan nhieu anh.

        Args:
            image_paths: Danh sach duong dan anh.
            top_k: So luong top predictions moi anh.

        Returns:
            List[Dict]: Danh sach ket qua du doan.

        Example:
            >>> results = predictor.batch_predict(["pho.jpg", "banh_mi.jpg"])
            >>> for r in results:
            ...     print(f"{r['display_name']}: {r['confidence']:.1%}")
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path, top_k=top_k)
                result["image_path"] = str(path)
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": str(path),
                    "status": "error",
                    "error": str(e),
                })
        return results


# ============================================================================
# CONVENIENCE FUNCTION CHO GRADIO (TV5)
# ============================================================================

# Global predictor instance (lazy load)
_global_predictor: Optional[FoodPredictor] = None


def quick_predict(
    image: Union[str, Path, Image.Image],
    checkpoint_path: Optional[Union[str, Path]] = None,
    top_k: int = 5,
    serving_grams: float = SERVING_GRAMS,
) -> Dict:
    """
    Ham tien ich 1 dong de du doan mon an.

    Tu dong load model lan dau, cache cho cac lan sau.
    Thiet ke cho TV5 tich hop vao Gradio app.

    Args:
        image: Anh can du doan (file path hoac PIL Image).
        checkpoint_path: Duong dan checkpoint. None = dung mac dinh.
        top_k: So luong top predictions.
        serving_grams: Khoi luong phan an (gram).

    Returns:
        Dict ket qua (giong FoodPredictor.predict()).

    Example:
        # Trong Gradio app (TV5):
        import gradio as gr
        from src.predict import quick_predict

        def predict_fn(image):
            result = quick_predict(image)
            label = f"{result['display_name']} ({result['confidence']:.1%})"
            calories = f"{result['nutrition']['serving_kcal']} kcal / {result['nutrition']['serving_grams']}g"
            return label, calories

        demo = gr.Interface(fn=predict_fn, inputs=gr.Image(type="pil"), outputs=["text", "text"])
    """
    global _global_predictor

    if _global_predictor is None:
        ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        _global_predictor = FoodPredictor(ckpt)

    return _global_predictor.predict(image, top_k=top_k, serving_grams=serving_grams)


# ============================================================================
# PRETTY PRINT
# ============================================================================


def print_prediction(result: Dict) -> None:
    """In ket qua du doan dep voi day du thong tin dinh duong."""
    print(f"\n{'='*60}")
    print(f" KET QUA DU DOAN")
    print(f"{'='*60}")
    print(f"  Mon an:     {result['display_name']}")
    print(f"  Do tin cay: {result['confidence']:.1%}")
    print(f"  Class ID:   {result['predicted_class']}")

    print(f"\n TOP PREDICTIONS:")
    for i, (cls, display, prob) in enumerate(result["top_k"], 1):
        bar = "" * int(prob * 30)
        print(f"  {i}. {display:20} {prob:.1%} {bar}")

    if result["nutrition"]["kcal_per_100g"] > 0:
        n = result["nutrition"]

        # Thong tin suat an thuc te (tu cong thuc nau an)
        real_w = n.get('real_serving_weight_g', 0)
        real_k = n.get('real_serving_kcal', 0)
        if real_w > 0:
            print(f"\n SUAT AN THUC TE (theo cong thuc):")
            print(f"  Khoi luong: {real_w:.0f}g")
            print(f"  Calories:   {real_k:.0f} kcal")

        # Macro nutrients
        print(f"\n DINH DUONG MACRO ({n['serving_grams']:.0f}g):")
        print(f"  Calories:    {n['serving_kcal']:.0f} kcal")
        print(f"  Protein:     {n['protein_g']:.1f}g")
        print(f"  Carb:        {n['carb_g']:.1f}g")
        print(f"  Fat:         {n['fat_g']:.1f}g")

        # Micro nutrients (neu co)
        has_micros = any(n.get(k, 0) > 0 for k in [
            'fiber_g', 'cholesterol_mg', 'calcium_mg', 'iron_mg'
        ])
        if has_micros:
            print(f"\n VI CHAT DINH DUONG ({n['serving_grams']:.0f}g):")
            if n.get('fiber_g', 0) > 0:
                print(f"  Chat xo:     {n['fiber_g']:.2f}g")
            if n.get('cholesterol_mg', 0) > 0:
                print(f"  Cholesterol: {n['cholesterol_mg']:.1f}mg")
            if n.get('calcium_mg', 0) > 0:
                print(f"  Canxi:       {n['calcium_mg']:.1f}mg")
            if n.get('phosphorus_mg', 0) > 0:
                print(f"  Photpho:     {n['phosphorus_mg']:.1f}mg")
            if n.get('iron_mg', 0) > 0:
                print(f"  Sat:         {n['iron_mg']:.2f}mg")
            if n.get('sodium_mg', 0) > 0:
                print(f"  Natri:       {n['sodium_mg']:.1f}mg")
            if n.get('potassium_mg', 0) > 0:
                print(f"  Kali:        {n['potassium_mg']:.1f}mg")
            if n.get('vitamin_a_mcg', 0) > 0:
                print(f"  Vitamin A:   {n['vitamin_a_mcg']:.1f}mcg")
            if n.get('vitamin_b1_mg', 0) > 0:
                print(f"  Vitamin B1:  {n['vitamin_b1_mg']:.3f}mg")
            if n.get('vitamin_c_mg', 0) > 0:
                print(f"  Vitamin C:   {n['vitamin_c_mg']:.2f}mg")

        print(f"  Nguon:       {n['source']}")

    if result["story"]:
        print(f"\n CAU CHUYEN:")
        print(f"  {result['story'][:120]}...")

    print(f"{'='*60}\n")


# ============================================================================
# MAIN (CLI)
# ============================================================================


def main():
    """Main function - chay tu command line."""
    parser = argparse.ArgumentParser(
        description="Du doan mon an Viet Nam tu anh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--image", type=str, required=True,
                        help="Duong dan toi anh can du doan")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="Duong dan checkpoint model")
    parser.add_argument("--top_k", type=int, default=5,
                        help="So luong top predictions")
    parser.add_argument("--serving", type=float, default=SERVING_GRAMS,
                        help="Khoi luong phan an (gram)")
    parser.add_argument("--gradcam", action="store_true",
                        help="Tao Grad-CAM heatmap")
    parser.add_argument("--gradcam_save", type=str, default=None,
                        help="Duong dan luu Grad-CAM image")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device")

    args = parser.parse_args()

    # Tao predictor
    predictor = FoodPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


    # Du doan
    if args.gradcam:
        save_path = args.gradcam_save or f"gradcam_{Path(args.image).stem}.png"
        result, heatmap = predictor.predict_with_gradcam(
            image_path=args.image,
            save_path=save_path,
            top_k=args.top_k,
        )
        print(f"[OK] Grad-CAM saved: {save_path}")
    else:
        result = predictor.predict(
            image=args.image,
            top_k=args.top_k,
            serving_grams=args.serving,
        )

    # In ket qua
    print_prediction(result)


if __name__ == "__main__":
    main()
