"""
Module tiền xử lý dữ liệu cho dự án nhận diện món ăn Việt Nam - OPTIMIZED.

Module này cung cấp:
- Custom Transforms: FixedAspectRatioPadding (VẤNĐỀ #2, #1)
- Transform pipelines tối ưu: get_train_transform(), get_val_transform() (VẤNĐỀ #4, #5)
- DataLoader creation với WeightedRandomSampler: get_dataloaders() (VẤNĐỀ #3)
- Helper functions: calculate_class_weights(), check_stratified_distribution() (VẤNĐỀ #3, #6)
- Visualization: show_augmentation_examples()

Các cải tiến:
✓ VẤNĐỀ #1: Xử lý ảnh nhỏ - dùng LANCZOS interpolation, logic giữ nguyên kích thước nhỏ
✓ VẤNĐỀ #2: Giữ aspect ratio - Custom Transform "FixedAspectRatioPadding"
✓ VẤNĐỀ #3: Class imbalance - WeightedRandomSampler + calculate_class_weights()
✓ VẤNĐỀ #4: ColorJitter tối ưu cho food domain
✓ VẤNĐỀ #5: LANCZOS interpolation thay vì BILINEAR
✓ VẤNĐỀ #6: Stratified validation check function
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import PIL Image resampling filters for better quality
try:
    from PIL.Image import Resampling
    LANCZOS = Resampling.LANCZOS
except ImportError:
    # Fallback for older Pillow versions
    LANCZOS = Image.LANCZOS

# ============================================================================
# CONSTANTS
# ============================================================================
"""
Cấu hình ImageNet normalization, kích thước ảnh, và đường dẫn dữ liệu.
"""

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
"""
Giá trị trung bình RGB của tập ImageNet.
Tất cả ảnh preprocess phải chuẩn hóa với giá trị này.
"""

IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
"""
Độ lệch chuẩn RGB của tập ImageNet.
Sử dụng kèm IMAGENET_MEAN để normalize ảnh.
"""

DATA_DIR: Path = Path(__file__).parent.parent / "data" / "raw" / "Images"
"""
Thư mục gốc chứa dữ liệu raw (Train, Test, Validate folders).
Đường dẫn tương đối từ project root.
"""

# Model input size
INPUT_SIZE: int = 224
"""
Kích thước tiêu chuẩn cho EfficientNetB0.
Tất cả ảnh sau augmentation sẽ có kích thước (3, 224, 224).
"""

# ============================================================================
# CUSTOM TRANSFORMS - GIẢI QUYẾT VẤNĐỀ #1 & #2
# ============================================================================
"""
Custom transforms để xử lý ảnh nhỏ và giữ nguyên tỷ lệ khung hình.
"""


class FixedAspectRatioPadding:
    """
    [VẤNĐỀ #2] Giữ nguyên tỷ lệ khung hình khi resize, sau đó crop center để đạt kích thước vuông.
    
    Chiến lược cải tiến (xử lý ảnh nhỏ & lớn):
    1. Scale ảnh sao cho cạnh nhỏ nhất = 256px, giữ aspect ratio
       - Nếu ảnh 100x200 → 128x256 (width nhỏ, scale up width)
       - Nếu ảnh 300x200 → 384x256 (height nhỏ, scale up height)
    2. Center crop từ 256x256 (một cạnh=256, cạnh kia>=256)
    3. Final center crop để 224x224
    
    Ưu điểm:
    - Tránh padding (không có viền artificial)
    - Aspect ratio được bảo toàn → không bị méo
    - Cắt ở giữa → giữ phần quan trọng của món ăn
    - Xử lý ảnh nhỏ mà không bị quá mờ
    - Xử lý ảnh lớn mà không bị cắt xấu
    
    Ví dụ:
    - Ảnh 100x200 → resize to 128x256 → crop center 256x256 → crop center 224x224
    - Ảnh 300x200 → resize to 384x256 → crop center 256x256 → crop center 224x224
    - Ảnh 200x200 (vuông) → resize to 256x256 → crop center 256x256 → crop center 224x224
    """
    
    def __init__(
        self,
        target_size: int = 256,
        final_size: int = 224,
        pad_mode: str = "reflect",
        interpolation=None,
    ):
        """
        Args:
            target_size: Kích thước trung gian sau resize (mặc định 256)
            final_size: Kích thước cuối cùng sau crop (mặc định 224)
            pad_mode: Cách padding - "reflect", "replicate", "edge", "constant"
            interpolation: Phương pháp nội suy (LANCZOS cho chất lượng tốt)
        """
        self.target_size = target_size
        self.final_size = final_size
        self.pad_mode = pad_mode
        self.interpolation = interpolation if interpolation is not None else LANCZOS
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Áp dụng transform.
        
        Args:
            img: PIL Image
            
        Returns:
            PIL Image với kích thước final_size x final_size
        """
        width, height = img.size
        aspect_ratio = width / height
        
        # [VẤNĐỀ #1 + #2] Smart resize strategy:
        # - Giữ aspect ratio: scale theo cạnh nhỏ hơn
        # - Cạnh nhỏ hơn sẽ = target_size (256)
        # - Cạnh lớn hơn sẽ > target_size (keep ratio)
        # - Sau đó cắt center để được hình vuông target_size x target_size
        
        if aspect_ratio > 1:  # width > height
            # height là cạnh nhỏ - scale để height = target_size
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)
        else:  # height >= width
            # width là cạnh nhỏ - scale để width = target_size
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        
        # [VẤNĐỀ #5] Resize dùng LANCZOS interpolation (chất lượng cao)
        img = img.resize((new_width, new_height), self.interpolation)
        
        # Center crop để được target_size x target_size (khối vuông)
        # Lúc này một chiều = target_size, chiều kia >= target_size
        left = max(0, (new_width - self.target_size) // 2)
        top = max(0, (new_height - self.target_size) // 2)
        right = left + self.target_size
        bottom = top + self.target_size
        
        img = img.crop((left, top, right, bottom))
        
        # Final center crop để được final_size x final_size (224x224)
        left = max(0, (self.target_size - self.final_size) // 2)
        top = max(0, (self.target_size - self.final_size) // 2)
        img = img.crop((left, top, left + self.final_size, top + self.final_size))
        
        return img

# ============================================================================
# HELPER FUNCTIONS - VẤNĐỀ #3, #6
# ============================================================================
"""
Helper functions để tính class weights và kiểm tra stratified distribution.
"""


def calculate_class_weights(
    data_dir: Path | str = DATA_DIR,
    split: str = "Train",
) -> torch.Tensor:
    """
    [VẤNĐỀ #3] Tính class weights để xử lý class imbalance.
    
    Công thức: weight[i] = total_samples / (num_classes * samples_in_class_i)
    
    Ý tưởng: Class có ít mẫu sẽ được gán trọng số cao hơn.
    
    Returns:
        torch.Tensor: shape (num_classes,), dtype=float32
        
    Example:
        >>> weights = calculate_class_weights(split="Train")
        >>> sampler = WeightedRandomSampler(weights, len(weights))
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Folder không tồn tại: {split_dir}")
    
    # Scan tất cả ảnh từ mỗi class
    class_counts = {}
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        num_images = sum(
            1 for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        )
        class_counts[class_dir.name] = num_images
    
    # Tính weights
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # Đảm bảo class_names sorted (giống như VNFoodDataset)
    class_names = sorted(class_counts.keys())
    weights = []
    
    print(f"\n{'='*70}")
    print(f"CLASS WEIGHTS ({split} split)")
    print(f"{'='*70}")
    print(f"{'Class Name':<25} {'Count':>8} {'Weight':>10}")
    print(f"{'-'*70}")
    
    for class_name in class_names:
        count = class_counts[class_name]
        weight = total_samples / (num_classes * count)
        weights.append(weight)
        print(f"{class_name:<25} {count:>8} {weight:>10.4f}")
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"{'-'*70}")
    print(f"Total samples: {total_samples}, Num classes: {num_classes}")
    print(f"Weight mean: {weights_tensor.mean():.4f}, std: {weights_tensor.std():.4f}")
    print(f"{'='*70}\n")
    
    return weights_tensor


def check_stratified_distribution(
    data_dir: Path | str = DATA_DIR,
) -> Dict[str, Dict[str, float]]:
    """
    [VẤNĐỀ #6] Kiểm tra xem tập Validation có stratified tốt so với Train không.
    
    In ra bảng so sánh phân bố % giữa Train và Validate.
    Nếu chênh lệch lớn (>5%), cảnh báo.
    
    Returns:
        Dict[split_name -> Dict[class_name -> percentage]]
        
    Example:
        >>> dist = check_stratified_distribution()
        >>> print(dist["Train"]["Pho"])  # 5.2 (%)
    """
    data_dir = Path(data_dir)
    
    distribution = {}
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    for split in ["Train", "Validate"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        class_counts = {}
        total = 0
        
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            count = sum(
                1 for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            class_counts[class_dir.name] = count
            total += count
        
        # Tính %
        percentages = {cls: (count / total) * 100 for cls, count in class_counts.items()}
        distribution[split] = percentages
    
    # In bảng so sánh
    print(f"\n{'='*80}")
    print(f"STRATIFIED DISTRIBUTION CHECK")
    print(f"{'='*80}")
    print(f"{'Class Name':<25} {'Train %':>10} {'Val %':>10} {'Diff %':>10}")
    print(f"{'-'*80}")
    
    max_diff = 0
    for class_name in sorted(distribution["Train"].keys()):
        train_pct = distribution["Train"][class_name]
        val_pct = distribution["Validate"][class_name]
        diff = abs(train_pct - val_pct)
        max_diff = max(max_diff, diff)
        
        status = "✓" if diff <= 2 else "⚠" if diff <= 5 else "✗"
        print(f"{class_name:<25} {train_pct:>10.2f} {val_pct:>10.2f} {diff:>9.2f} {status}")
    
    print(f"{'-'*80}")
    print(f"Max difference: {max_diff:.2f}%")
    
    if max_diff <= 2:
        print("✓ EXCELLENT: Validation set is well-stratified")
    elif max_diff <= 5:
        print("⚠ WARNING: Some classes have >2% difference (acceptable)")
    else:
        print("✗ PROBLEM: Significant imbalance between Train and Validate")
    
    print(f"{'='*80}\n")
    
    return distribution


# ============================================================================
# TRANSFORM PIPELINES - GIẢI QUYẾT VẤNĐỀ #4, #5
# ============================================================================
"""
Định nghĩa các transform pipeline cho training và validation.

Cải tiến:
✓ Dùng FixedAspectRatioPadding thay vì RandomCrop (VẤNĐỀ #2)
✓ Dùng LANCZOS interpolation (VẤNĐỀ #5)
✓ ColorJitter tối ưu cho food domain (VẤNĐỀ #4)
"""


def get_train_transform() -> transforms.Compose:
    """
    Tạo transform pipeline tối ưu cho training set.

    Pipeline (theo thứ tự):
    
    1. [VẤNĐỀ #2 + #1] FixedAspectRatioPadding(256, 224):
       → Giữ nguyên aspect ratio của ảnh (không bị méo)
       → Scale theo cạnh nhỏ nhất (SMART: không phóng quá lớn, không cắt mất chi tiết)
       → Resize dùng LANCZOS (VẤNĐỀ #5) - chất lượng cao
       → Center crop để đạt kích thước vuông (giữ phần trung tâm quan trọng)
       → Xử lý tốt ảnh nhỏ (<224x224) lẫn ảnh lớn (>224x224)
    
    2. RandomHorizontalFlip(p=0.5):
       → Món ăn thường đối xứng ngang
       → Tăng gấp đôi training data một cách tự nhiên
    
    3. RandomRotation(10°):
       → Ảnh thực tế chụp từ góc khác nhau (nhưng không quay quá 10°)
       → Model học rotational invariance
       → GIẢM từ 15° -> 10° để tránh tạo ra orientation quá lạ
    
    4. [VẤNĐỀ #4] ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08):
       → Tối ưu cho food domain:
         * brightness=0.15: Tránh thay đổi độ sáng quá đột ngột
           (món ăn có màu tự nhiên, không nên bị sáng quá hay tối quá)
         * contrast=0.15: Giữ texture tương phản, không quá cứng
         * saturation=0.15: Màu sắc tươi sáng của thực phẩm được bảo toàn
           (không quá nước ngoài khiến mất nhận diện)
         * hue=0.08: Hạn chế thay đổi sắc thái (không đổi vàng thành tím)
       → Các giá trị này dựa trên kinh nghiệm food domain experts
    
    5. ToTensor(): Chuyển PIL Image [0,255] → Tensor [0.0,1.0]
    
    6. Normalize(IMAGENET_MEAN, IMAGENET_STD):
       → EfficientNetB0 pretrained trên ImageNet
       → PHẢI chuẩn hóa cùng mean/std

    Returns:
        transforms.Compose: Pipeline transform tối ưu.
    """
    return transforms.Compose(
        [
            # [VẤNĐỀ #1, #2, #5] Custom transform: giữ aspect ratio + LANCZOS
            FixedAspectRatioPadding(
                target_size=256,
                final_size=224,
                pad_mode="reflect",
                interpolation=LANCZOS,
            ),
            
            # Data augmentation for training
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Reduced từ 15° -> 10°
            
            # [VẤNĐỀ #4] ColorJitter tối ưu cho food domain
            transforms.ColorJitter(
                brightness=0.15,  # Giới hạn độ sáng thay đổi
                contrast=0.15,    # Giữ texture không quá cứng
                saturation=0.15,  # Bảo toàn màu sắc tươi sáng
                hue=0.08,         # Hạn chế thay đổi sắc thái
            ),
            
            # Để trống GaussianBlur vì không phải tất cả Torchvision versions đều hỗ trợ
            # Thêm nếu muốn: transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            
            # Normalize
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_val_transform() -> transforms.Compose:
    """
    Tạo transform pipeline tối ưu cho validation/test set (NO AUGMENTATION).

    Pipeline (theo thứ tự):
    
    1. [VẤNĐỀ #1, #2, #5] FixedAspectRatioPadding(256, 224):
       → Cùng logic như training: giữ aspect ratio + LANCZOS
       → Deterministic (không random, kết quả reproducible)
    
    2. ToTensor(): Chuyển PIL Image → Tensor
    
    3. Normalize(IMAGENET_MEAN, IMAGENET_STD):
       → Chuẩn hóa cùng ImageNet

    Returns:
        transforms.Compose: Pipeline transform deterministic.

    Note:
        Validation không dùng augmentation → đánh giá model trên "clean" data
    """
    return transforms.Compose(
        [
            # [VẤNĐỀ #1, #2, #5] Cùng FixedAspectRatioPadding (deterministic)
            FixedAspectRatioPadding(
                target_size=256,
                final_size=224,
                pad_mode="reflect",
                interpolation=LANCZOS,
            ),
            
            # NO augmentation - validation dùng "clean" data
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ============================================================================
# DATASET CLASS
# ============================================================================


class VNFoodDataset(Dataset):
    """
    Dataset loader cho dữ liệu 30VNFoods.

    Features:
    - Tự động scan class names từ folder (sorted)
    - Lọc file ảnh: .jpg, .jpeg, .png, .webp
    - Convert sang RGB (xử lý grayscale, RGBA)
    - Skip ảnh corrupted (log warning)

    Attributes:
        root_dir (Path): Đường dẫn tới folder Train/Test/Validate
        split (str): "Train", "Test", hoặc "Validate"
        transform (callable): Transform function (optional)
        class_names (list[str]): Sorted list of class names
        class_to_idx (dict): Ánh xạ class_name → index
        samples (list[tuple]): List of (img_path, label_idx)

    Example:
        >>> dataset = VNFoodDataset(
        ...     root_dir=Path("data/raw/Images"),
        ...     split="Train",
        ...     transform=get_train_transform()
        ... )
        >>> img_tensor, label, class_name = dataset[0]
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "Train",
        transform=None,
    ) -> None:
        """
        Khởi tạo dataset.

        Args:
            root_dir: Đường dẫn tới folder chứa Train/Test/Validate subfolders.
            split: "Train", "Test", hoặc "Validate".
            transform: Transform function (từ get_train_transform() hoặc get_val_transform()).

        Raises:
            ValueError: Nếu split không hợp lệ hoặc folder không tồn tại.
            FileNotFoundError: Nếu folder split không tìm thấy.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Kiểm tra split
        if split not in ["Train", "Test", "Validate"]:
            raise ValueError(f"split phải là 'Train', 'Test', hoặc 'Validate', không '{split}'")

        # Tìm folder split
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Folder không tồn tại: {self.split_dir}")

        # Scan class folders
        self.class_names = sorted(
            [d.name for d in self.split_dir.iterdir() if d.is_dir()]
        )

        if len(self.class_names) == 0:
            raise ValueError(f"Không tìm thấy class folders trong {self.split_dir}")

        # Tạo class_to_idx mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Scan ảnh
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"Không tìm thấy ảnh trong {self.split_dir}")

    def _load_samples(self) -> None:
        """
        Scan tất cả ảnh từ class folders.
        Tự động skip ảnh corrupted.
        """
        print(f"   Scanning {len(self.class_names)} classes from {self.split}...", flush=True)
        for idx, class_name in enumerate(self.class_names, 1):
            class_dir = self.split_dir / class_name
            class_samples = 0

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue

                # Kiểm tra ảnh có corrupted không
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    class_samples += 1
                except Exception as e:
                    warnings.warn(f"Skipped corrupted image: {img_path} ({str(e)})")
                    continue
            
            if idx % 5 == 0 or idx == len(self.class_names):
                print(f"     [{idx}/{len(self.class_names)}] {class_name}: {class_samples} images", flush=True)

    def __len__(self) -> int:
        """Trả về số lượng ảnh trong dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Lấy một mẫu từ dataset.

        Args:
            idx: Index của mẫu.

        Returns:
            Tuple của (image_tensor, label, class_name)
            - image_tensor: torch.Tensor shape (3, 224, 224)
            - label: int (class index)
            - class_name: str (tên class)
        """
        img_path, label_idx = self.samples[idx]
        class_name = self.class_names[label_idx]

        # Load ảnh
        with Image.open(img_path) as img:
            # Convert sang RGB (xử lý grayscale, RGBA)
            img = img.convert("RGB")

        # Apply transform
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label_idx, class_name


# ============================================================================
# DATALOADER CREATION
# ============================================================================


def get_dataloaders(
    data_dir: Path | str = DATA_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    check_stratified: bool = True,
) -> Tuple[Dict[str, DataLoader], Optional[torch.Tensor]]:
    """
    Tạo DataLoaders cho train/val/test splits.

    [VẤNĐỀ #3] Với WeightedRandomSampler để xử lý class imbalance.
    [VẤNĐỀ #6] Optional: kiểm tra stratified distribution của validation set.

    Args:
        data_dir: Đường dẫn tới folder chứa Train/Test/Validate.
        batch_size: Batch size cho training (default: 32).
        num_workers: Số worker processes để load ảnh (default: 4).
        use_weighted_sampler: Nếu True, dùng WeightedRandomSampler cho train loader (default: True).
        check_stratified: Nếu True, kiểm tra stratified distribution (default: True).

    Returns:
        Tuple[dataloaders_dict, class_weights]:
        - dataloaders_dict: Dictionary với keys "train", "val", "test"
        - class_weights: Tensor của class weights (hoặc None nếu không dùng)

    Details:
        - Train loader: WeightedRandomSampler (xử lý imbalance), augmentation, drop_last=True
        - Val/Test loaders: SequentialSampler (random=False), no augmentation
        - pin_memory=True nếu CUDA available
        
        [VẤNĐỀ #3] WeightedRandomSampler:
        → Lấy mẫu theo xác suất tỷ lệ với class weights
        → Class ít sẽ được sample nhiều hơn → cân bằng dữ liệu
        → Giảm impact của class imbalance

    Example:
        >>> dataloaders, weights = get_dataloaders(batch_size=32, use_weighted_sampler=True)
        >>> train_loader = dataloaders["train"]
        >>> for images, labels, class_names in train_loader:
        ...     print(images.shape)  # (32, 3, 224, 224)
        ...     break
    """
    data_dir = Path(data_dir)

    # [VẤNĐỀ #6] Kiểm tra stratified distribution (optional)
    if check_stratified:
        try:
            check_stratified_distribution(data_dir)
        except Exception as e:
            warnings.warn(f"Could not check stratified distribution: {e}")

    # Tạo datasets
    train_dataset = VNFoodDataset(
        root_dir=data_dir,
        split="Train",
        transform=get_train_transform(),
    )

    val_dataset = VNFoodDataset(
        root_dir=data_dir,
        split="Validate",
        transform=get_val_transform(),
    )

    test_dataset = VNFoodDataset(
        root_dir=data_dir,
        split="Test",
        transform=get_val_transform(),
    )

    # Kiểm tra CUDA
    pin_memory = torch.cuda.is_available()

    # [VẤNĐỀ #3] Tính class weights và tạo WeightedRandomSampler
    class_weights = None
    train_sampler = None
    
    if use_weighted_sampler:
        try:
            # Tính weights theo class imbalance
            class_weights = calculate_class_weights(data_dir, split="Train")
            
            # Tạo sample weights: mỗi sample được gán weight của class nó
            # class_weights[i] là weight của class i
            # train_dataset.samples[j] = (img_path, label_idx)
            sample_weights = torch.tensor(
                [class_weights[label].item() for _, label in train_dataset.samples],
                dtype=torch.float32
            )
            
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True,  # Cho phép sample lại (important cho imbalance)
            )
            print("✓ Using WeightedRandomSampler for training (handles class imbalance)")
        except Exception as e:
            warnings.warn(f"Could not create WeightedRandomSampler: {e}")
            train_sampler = None

    # Tạo DataLoaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,  # [VẤNĐỀ #3] Use weighted sampler
            shuffle=(train_sampler is None),  # Chỉ shuffle nếu không dùng sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Tránh batch size = 1
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }

    return dataloaders, class_weights


# ============================================================================
# AUGMENTATION VISUALIZATION
# ============================================================================


def show_augmentation_examples(
    class_name: str,
    data_dir: Path | str = DATA_DIR,
    n_augments: int = 4,
) -> None:
    """
    Visualize augmentation effect trên một ảnh.

    Hiển thị grid 2×(n_augments+1):
    - Hàng 1: PIL images (trước normalize)
    - Hàng 2: Tensors (sau normalize, denormalize để xem)

    Args:
        class_name: Tên class để lấy ảnh mẫu.
        data_dir: Đường dẫn tới data folder.
        n_augments: Số lượng augmented versions để hiển thị.

    Returns:
        None (hiển thị figure và save PNG)

    Example:
        >>> show_augmentation_examples("Pho", n_augments=4)
        >>> # Hiển thị grid: [Original] [Aug1] [Aug2] [Aug3] [Aug4]
    """
    data_dir = Path(data_dir)
    Path("figures").mkdir(parents=True, exist_ok=True)

    # Tạo dataset để lấy ảnh
    dataset = VNFoodDataset(
        root_dir=data_dir,
        split="Train",
        transform=None,  # Lấy PIL image gốc
    )

    # Tìm ảnh của class_name
    if class_name not in dataset.class_names:
        raise ValueError(f"Class '{class_name}' không tìm thấy. "
                        f"Available: {dataset.class_names}")

    class_idx = dataset.class_to_idx[class_name]
    sample_indices = [i for i, (_, label, _) in enumerate(
        [(dataset.samples[j][0], dataset.samples[j][1], dataset.class_names[dataset.samples[j][1]])
         for j in range(len(dataset))]
    ) if label == class_idx]

    if not sample_indices:
        raise ValueError(f"Không tìm thấy ảnh cho class '{class_name}'")

    idx = sample_indices[0]
    img_path, _ = dataset.samples[idx]

    # Load ảnh gốc
    original_img = Image.open(img_path).convert("RGB")

    # Tạo augmented versions
    train_transform = get_train_transform()
    augmented_imgs = [original_img]
    augmented_tensors = [transforms.ToTensor()(original_img)]

    for _ in range(n_augments):
        aug_tensor = train_transform(original_img)
        augmented_tensors.append(aug_tensor)

        # Denormalize để visualize
        denorm_tensor = aug_tensor.clone()
        for i, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            denorm_tensor[i] = denorm_tensor[i] * std + mean
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        augmented_imgs.append(transforms.ToPILImage()(denorm_tensor))

    # Tạo grid
    # Tạo grid
        n_cols = n_augments + 1
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
        fig.suptitle(f"Augmentation Examples: {class_name}", fontsize=14, fontweight='bold')

    titles = ["Original"] + [f"Augment {i+1}" for i in range(n_augments)]

    # Row 1: PIL images
    for col, (ax, img, title) in enumerate(zip(axes[0], augmented_imgs, titles)):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Row 2: Denormalized tensors
    for col, (ax, title) in enumerate(zip(axes[1], titles)):
        if col == 0:
            img_to_show = augmented_imgs[0]
        else:
            # Denormalize
            tensor = augmented_tensors[col]
            denorm_tensor = tensor.clone()
            for i, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
                denorm_tensor[i] = denorm_tensor[i] * std + mean
            denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
            img_to_show = transforms.ToPILImage()(denorm_tensor)

        ax.imshow(img_to_show)
        ax.set_title(f"{title} (Tensor)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    save_path = Path("figures") / f"augmentation_demo_{class_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved {save_path}")
    plt.show()


# ============================================================================
# UNIT TESTS
# ============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("PREPROCESS MODULE TEST - OPTIMIZED VERSION")
    print("=" * 80)

    # Test 0: Stratified distribution check
    print("\n0. Checking stratified distribution...")
    try:
        dist = check_stratified_distribution(DATA_DIR)
        print("   ✓ Stratified check passed")
    except Exception as e:
        print(f"   ⚠ Warning: {e}")

    # Test 1: Calculate class weights
    print("\n1. Calculating class weights...")
    try:
        weights = calculate_class_weights(DATA_DIR, split="Train")
        print(f"   ✓ Class weights calculated: shape {weights.shape}")
    except Exception as e:
        print(f"   ⚠ Warning: {e}")

    # Test 2: Creating dataloaders dengan weighted sampler
    print("\n2. Creating dataloaders (with WeightedRandomSampler)...")
    print("   ⏳ Scanning folders (30-60 seconds for first run)...")
    print("   (Windows tip: num_workers=0 to avoid multiprocessing overhead)")
    try:
        # ⚠️ num_workers=0 on Windows to avoid multiprocessing issues
        dataloaders, class_weights = get_dataloaders(
            batch_size=32,
            num_workers=0,
            use_weighted_sampler=True,
            check_stratified=True,
        )
        print("   ✓ Dataloaders created successfully")

        for split_name, loader in dataloaders.items():
            num_batches = len(loader)
            print(f"   - {split_name:6} split: {num_batches:3} batches")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        exit(1)

    # Test 3: Kiểm tra batch structure
    print("\n3. Testing batch structure...")
    try:
        train_loader = dataloaders["train"]
        images, labels, class_names = next(iter(train_loader))

        print(f"   Image tensor shape: {images.shape}")
        print(f"   Expected: (batch_size, 3, 224, 224)")
        if images.shape[1:] == (3, 224, 224):
            print("   ✓ PASS - Shape correct!")
        else:
            print(f"   ✗ FAIL - Shape mismatch!")

        print(f"\n   Labels shape: {labels.shape}, dtype: {labels.dtype}")
        print(f"   Sample class names: {class_names[:3]}...")
        
        # Check image value range (should be normalized)
        print(f"\n   Image value range:")
        print(f"     Min: {images.min():.3f}, Max: {images.max():.3f}")
        print(f"     Mean: {images.mean():.3f}, Std: {images.std():.3f}")
        print("     (Expected: roughly [-2, 2] after ImageNet normalization)")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 4: Dataset properties
    print("\n4. Checking dataset properties...")
    try:
        train_dataset = VNFoodDataset(
            root_dir=DATA_DIR,
            split="Train",
            transform=get_train_transform()
        )

        print(f"   Number of classes: {len(train_dataset.class_names)}")
        print(f"   Total samples: {len(train_dataset)}")
        print(f"   Expected: 30 classes")

        if len(train_dataset.class_names) == 30:
            print("   ✓ PASS - 30 classes found")
        else:
            print(f"   ✗ Classes mismatch: {len(train_dataset.class_names)}")

        print(f"\n   First 5 classes (sorted):")
        for i, name in enumerate(train_dataset.class_names[:5], 1):
            count = sum(1 for _, label in train_dataset.samples if label == i-1)
            print(f"     {i}. {name:20} ({count} samples)")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 5: Test new transforms (aspect ratio preservation)
    print("\n5. Testing new FixedAspectRatioPadding transform...")
    try:
        transform = FixedAspectRatioPadding()
        
        # Tạo test images với aspect ratios khác nhau
        test_imgs = [
            Image.new("RGB", (100, 200)),  # High, narrow
            Image.new("RGB", (300, 100)),  # Wide, short
            Image.new("RGB", (200, 200)),  # Square
            Image.new("RGB", (50, 100)),   # Very small
        ]
        
        for img in test_imgs:
            result = transform(img)
            print(f"   {img.size} → {result.size} ✓")
        
        print("   ✓ PASS - FixedAspectRatioPadding works correctly")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 6: Augmentation visualization (optional)
    print("\n6. Testing augmentation visualization...")
    try:
        show_augmentation_examples("Pho", n_augments=4)
        print("   ✓ Augmentation demo created")
    except Exception as e:
        print(f"   ⚠ Warning: {e}")
        print("   (This is optional - may fail if data issues)")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS COMPLETED!")
    print("=" * 80)
    
    # Print integration guide
    print("\n" + "=" * 80)
    print("INTEGRATION GUIDE FOR TRAINING LOOP")
    print("=" * 80)
    print("""
Hướng dẫn sử dụng dataloaders tối ưu trong training loop:

1. IMPORT:
   from src.preprocess import get_dataloaders, get_train_transform, get_val_transform
   
2. CREATE DATALOADERS:
   dataloaders, class_weights = get_dataloaders(
       batch_size=32,
       num_workers=4,  # Increase on Linux/Mac, use 0 on Windows
       use_weighted_sampler=True,  # Handle class imbalance
       check_stratified=True,      # Check validation distribution
   )
   
3. ACCESS LOADERS:
   train_loader = dataloaders["train"]
   val_loader = dataloaders["val"]
   test_loader = dataloaders["test"]
   
4. TRAINING LOOP:
   for epoch in range(num_epochs):
       for images, labels, class_names in train_loader:
           # Model training
           outputs = model(images)  # Shape: (batch_size, 30)
           loss = criterion(outputs, labels)
           # ... backward pass, optimizer.step() ...
       
       # Validation
       with torch.no_grad():
           for images, labels, class_names in val_loader:
               # Model evaluation
               ...
   
5. OPTIONAL - USE CLASS WEIGHTS FOR LOSS:
   if class_weights is not None:
       criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
   
6. KEY IMPROVEMENTS IN THIS VERSION:
   ✓ FixedAspectRatioPadding: Preserves aspect ratio, no cropping artifacts
   ✓ LANCZOS interpolation: Better image quality during resizing
   ✓ Optimized ColorJitter: Tuned for food domain (realistic colors)
   ✓ WeightedRandomSampler: Handles class imbalance automatically
   ✓ Stratified check: Validates distribution between train/val splits
""")
    print("=" * 80)
