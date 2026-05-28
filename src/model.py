"""
Module dinh nghia kien truc model cho du an nhan dien mon an Viet Nam.

Module nay cung cap:
- create_model(): Tao model EfficientNetB0 / MobileNetV3 pretrained
- unfreeze_backbone(): Mo dan cac layer de fine-tune
- get_model_info(): In thong tin so luong params
- GradCAM: Visualization giai thich du doan cua model
- get_target_layer(): Tu dong tim layer cuoi cho Grad-CAM

Thiet ke:
- Classifier head: Dropout(0.3) -> Linear(in, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 30)
- 2-phase training: Phase 1 freeze backbone, Phase 2 unfreeze fine-tune
- Grad-CAM hook vao last conv layer de tao heatmap
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ============================================================================
# CONSTANTS
# ============================================================================

SUPPORTED_MODELS: List[str] = ["efficientnet_b0", "mobilenet_v3_small"]
"""Danh sach model duoc ho tro."""

NUM_CLASSES: int = 29
"""So luong class mon an Viet Nam (da loai Banh duc - noisy class)."""


# ============================================================================
# MODEL CREATION
# ============================================================================


def create_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Tao model phan loai mon an Viet Nam voi pretrained backbone.

    Backbone duoc load tu ImageNet pretrained weights, sau do thay the
    classifier head bang custom head phu hop voi 30 classes.

    Args:
        model_name: Ten model ("efficientnet_b0" hoac "mobilenet_v3_small").
        num_classes: So luong classes (mac dinh 30).
        pretrained: Neu True, dung ImageNet pretrained weights.
        freeze_backbone: Neu True, dong bang tat ca backbone params.

    Returns:
        nn.Module: Model da config san sang train.

    Raises:
        ValueError: Neu model_name khong duoc ho tro.

    Example:
        >>> model = create_model("efficientnet_b0", num_classes=30, freeze_backbone=True)
        >>> output = model(torch.randn(1, 3, 224, 224))
        >>> print(output.shape)  # (1, 30)
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' khong duoc ho tro. "
            f"Chon: {SUPPORTED_MODELS}"
        )

    if model_name == "efficientnet_b0":
        # EfficientNetB0 pretrained ImageNet
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # Lay so features cua classifier goc
        in_features = model.classifier[1].in_features  # 1280

        # Thay classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    elif model_name == "mobilenet_v3_small":
        # MobileNetV3-Small pretrained ImageNet
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)

        # Lay so features cua classifier goc
        in_features = model.classifier[0].in_features  # 576

        # Thay classifier head
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(256, num_classes),
        )

    # Freeze backbone neu can
    if freeze_backbone:
        _freeze_backbone(model, model_name)

    print(f"✓ Model '{model_name}' da tao thanh cong (num_classes={num_classes})")
    if freeze_backbone:
        print(f"  → Backbone FROZEN - chi train classifier head")

    return model


def _freeze_backbone(model: nn.Module, model_name: str) -> None:
    """
    Dong bang tat ca backbone params (chi giu classifier trainable).

    Args:
        model: Model can freeze.
        model_name: Ten model de xac dinh backbone.
    """
    if model_name == "efficientnet_b0":
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_name == "mobilenet_v3_small":
        for param in model.features.parameters():
            param.requires_grad = False


def unfreeze_backbone(
    model: nn.Module,
    model_name: str,
    num_layers: Optional[int] = None,
) -> None:
    """
    Mo (unfreeze) cac layer backbone de fine-tune.

    Chien luoc: Mo dan tu cuoi len (cac layer gan output hoc features
    cao cap hon, phu hop voi fine-tune).

    Args:
        model: Model can unfreeze.
        model_name: Ten model.
        num_layers: So layer mo tu cuoi len. None = mo tat ca.

    Example:
        >>> model = create_model("efficientnet_b0", freeze_backbone=True)
        >>> # Train classifier head 5 epochs...
        >>> unfreeze_backbone(model, "efficientnet_b0", num_layers=3)
        >>> # Fine-tune 3 block cuoi + classifier...
    """
    if model_name == "efficientnet_b0":
        backbone = model.features
    elif model_name == "mobilenet_v3_small":
        backbone = model.features
    else:
        raise ValueError(f"Model '{model_name}' khong duoc ho tro.")

    # Lay danh sach children cua backbone
    children = list(backbone.children())
    total_layers = len(children)

    if num_layers is None:
        # Mo tat ca
        for param in backbone.parameters():
            param.requires_grad = True
        print(f"✓ Unfreeze ALL {total_layers} backbone layers")
    else:
        # Mo N layer cuoi
        num_layers = min(num_layers, total_layers)
        layers_to_unfreeze = children[-num_layers:]

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"✓ Unfreeze {num_layers}/{total_layers} backbone layers (tu cuoi)")


# ============================================================================
# MODEL INFO
# ============================================================================


def get_model_info(model: nn.Module) -> Dict[str, int]:
    """
    In thong tin so luong parameters cua model.

    Args:
        model: Model can kiem tra.

    Returns:
        Dict voi keys: total, trainable, frozen.

    Example:
        >>> info = get_model_info(model)
        >>> print(f"Trainable: {info['trainable']:,}")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\n{'='*50}")
    print(f"MODEL PARAMETERS INFO")
    print(f"{'='*50}")
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable params: {trainable:>12,}")
    print(f"  Frozen params:    {frozen:>12,}")
    print(f"  Trainable ratio:  {trainable/total*100:>11.1f}%")
    print(f"{'='*50}\n")

    return {"total": total, "trainable": trainable, "frozen": frozen}


# ============================================================================
# GRAD-CAM
# ============================================================================


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Tu dong tim layer cuoi cung cua backbone de dung cho Grad-CAM.

    Grad-CAM can hook vao conv layer cuoi cung (truoc pooling/classifier)
    de lay activation maps va gradients.

    Args:
        model: Model da tao.
        model_name: Ten model.

    Returns:
        nn.Module: Layer cuoi cung cua backbone.

    Example:
        >>> target_layer = get_target_layer(model, "efficientnet_b0")
    """
    if model_name == "efficientnet_b0":
        # EfficientNetB0: features[-1] la block cuoi (conv + bn + silu)
        return model.features[-1]
    elif model_name == "mobilenet_v3_small":
        # MobileNetV3: features[-1] la ConvBNActivation cuoi
        return model.features[-1]
    else:
        raise ValueError(f"Model '{model_name}' khong duoc ho tro.")


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping).

    Tao heatmap giai thich vung anh nao model chu y nhat khi du doan.
    Huu ich de kiem tra model co "nhin" dung vao mon an hay khong.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.

    Attributes:
        model: Model da train.
        target_layer: Layer de hook gradients.
        activations: Feature maps tu forward pass.
        gradients: Gradients tu backward pass.

    Example:
        >>> gradcam = GradCAM(model, get_target_layer(model, "efficientnet_b0"))
        >>> heatmap = gradcam.generate(input_tensor, target_class=0)
        >>> gradcam.visualize("test.jpg", model, transform, class_names)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Khoi tao Grad-CAM voi model va target layer.

        Args:
            model: Model da train (phai o eval mode khi generate).
            target_layer: Conv layer cuoi cung cua backbone.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Dang ky hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        """Hook luu activation maps tu forward pass."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        """Hook luu gradients tu backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Tao Grad-CAM heatmap cho 1 input tensor.

        Args:
            input_tensor: Tensor shape (1, 3, 224, 224), da normalized.
            target_class: Class index de giai thich. None = class co xac suat cao nhat.

        Returns:
            np.ndarray: Heatmap shape (224, 224), gia tri [0, 1].
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass cho target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Tinh Grad-CAM
        # gradients: (1, C, H, W) -> weights: (C,)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global Average Pooling

        # Weighted combination: sum(weights * activations)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Chi lay gia tri duong

        # Resize ve 224x224
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def visualize(
        self,
        image_path: Union[str, Path],
        transform: transforms.Compose,
        class_names: List[str],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> Tuple[np.ndarray, str, float]:
        """
        Tao va hien thi Grad-CAM overlay len anh goc.

        Args:
            image_path: Duong dan toi file anh.
            transform: Transform pipeline (val transform).
            class_names: Danh sach ten cac class.
            save_path: Duong dan luu anh. None = khong luu.
            show: Neu True, hien thi figure.

        Returns:
            Tuple[heatmap, predicted_class_name, confidence]:
            - heatmap: np.ndarray (224, 224)
            - predicted_class_name: str
            - confidence: float
        """
        # Load va transform anh
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        # Chuyen sang device cua model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Forward + Grad-CAM
        with torch.enable_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            predicted_idx = probs.argmax(dim=1).item()
            confidence = probs[0, predicted_idx].item()
            predicted_name = class_names[predicted_idx]

            # Generate heatmap
            heatmap = self.generate(input_tensor, target_class=predicted_idx)

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Anh goc
        img_resized = img.resize((224, 224))
        axes[0].imshow(img_resized)
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[1].axis("off")

        # Overlay
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        heatmap_colored = cm.jet(heatmap)[:, :, :3]  # RGB tu colormap
        overlay = 0.6 * img_array + 0.4 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Overlay: {predicted_name} ({confidence:.1%})", fontsize=12
        )
        axes[2].axis("off")

        plt.suptitle("Grad-CAM Visualization", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Grad-CAM saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return heatmap, predicted_name, confidence

    def remove_hooks(self) -> None:
        """Go bo hooks khi khong can dung nua."""
        self._forward_hook.remove()
        self._backward_hook.remove()
        print("✓ Grad-CAM hooks removed")


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: Model Module")
    print("=" * 70)

    # Test 1: Tao EfficientNetB0
    print("\n1. Tao EfficientNetB0 (freeze backbone)...")
    try:
        model_eff = create_model("efficientnet_b0", freeze_backbone=True)
        info_eff = get_model_info(model_eff)

        # Kiem tra output shape
        dummy = torch.randn(2, 3, 224, 224)
        out = model_eff(dummy)
        assert out.shape == (2, 30), f"Shape sai: {out.shape}"
        print(f"   ✓ Output shape: {out.shape}")
    except Exception as e:
        print(f"   ✗ Loi: {e}")

    # Test 2: Tao MobileNetV3
    print("\n2. Tao MobileNetV3-Small (freeze backbone)...")
    try:
        model_mob = create_model("mobilenet_v3_small", freeze_backbone=True)
        info_mob = get_model_info(model_mob)

        out = model_mob(dummy)
        assert out.shape == (2, 30), f"Shape sai: {out.shape}"
        print(f"   ✓ Output shape: {out.shape}")
    except Exception as e:
        print(f"   ✗ Loi: {e}")

    # Test 3: Unfreeze backbone
    print("\n3. Unfreeze 3 layers cuoi cua EfficientNetB0...")
    try:
        unfreeze_backbone(model_eff, "efficientnet_b0", num_layers=3)
        info_after = get_model_info(model_eff)
        assert info_after["trainable"] > info_eff["trainable"]
        print(f"   ✓ Trainable tang: {info_eff['trainable']:,} -> {info_after['trainable']:,}")
    except Exception as e:
        print(f"   ✗ Loi: {e}")

    # Test 4: Grad-CAM
    print("\n4. Test Grad-CAM...")
    try:
        model_eff.eval()
        target_layer = get_target_layer(model_eff, "efficientnet_b0")
        gradcam = GradCAM(model_eff, target_layer)

        dummy_input = torch.randn(1, 3, 224, 224)
        heatmap = gradcam.generate(dummy_input)
        assert heatmap.shape == (224, 224), f"Heatmap shape sai: {heatmap.shape}"
        assert 0 <= heatmap.min() and heatmap.max() <= 1, "Heatmap khong normalize"
        print(f"   ✓ Heatmap shape: {heatmap.shape}, range: [{heatmap.min():.2f}, {heatmap.max():.2f}]")

        gradcam.remove_hooks()
    except Exception as e:
        print(f"   ✗ Loi: {e}")

    # Test 5: Model khong hop le
    print("\n5. Test model khong hop le...")
    try:
        create_model("resnet50")
        print("   ✗ Khong raise ValueError!")
    except ValueError as e:
        print(f"   ✓ ValueError: {e}")

    print("\n" + "=" * 70)
    print("✓ TAT CA TEST PASSED!")
    print("=" * 70)
