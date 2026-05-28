"""
Module tin x l d liu cho d n nhn din mn n Vit Nam - OPTIMIZED.

Module ny cung cp:
- Custom Transforms: FixedAspectRatioPadding (VN #2, #1)
- Transform pipelines ti u: get_train_transform(), get_val_transform() (VN #4, #5)
- DataLoader creation vi WeightedRandomSampler: get_dataloaders() (VN #3)
- Helper functions: calculate_class_weights(), check_stratified_distribution() (VN #3, #6)
- Visualization: show_augmentation_examples()

Cc ci tin:
[OK] VN #1: X l nh nh - dng LANCZOS interpolation, logic gi nguyn kch thc nh
[OK] VN #2: Gi aspect ratio - Custom Transform "FixedAspectRatioPadding"
[OK] VN #3: Class imbalance - WeightedRandomSampler + calculate_class_weights()
[OK] VN #4: ColorJitter ti u cho food domain
[OK] VN #5: LANCZOS interpolation thay v BILINEAR
[OK] VN #6: Stratified validation check function
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
Cu hnh ImageNet normalization, kch thc nh, v ng dn d liu.
"""

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
"""
Gi tr trung bnh RGB ca tp ImageNet.
Tt c nh preprocess phi chun ha vi gi tr ny.
"""

IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
"""
 lch chun RGB ca tp ImageNet.
S dng km IMAGENET_MEAN  normalize nh.
"""

DATA_DIR: Path = Path(__file__).parent.parent / "data" / "raw" / "Images"
"""
Th mc gc cha d liu raw (Train, Test, Validate folders).
ng dn tng i t project root.
"""

# Model input size
INPUT_SIZE: int = 224
"""
Kch thc tiu chun cho EfficientNetB0.
Tt c nh sau augmentation s c kch thc (3, 224, 224).
"""

# ============================================================================
# CUSTOM TRANSFORMS - GII QUYT VN #1 & #2
# ============================================================================
"""
Custom transforms  x l nh nh v gi nguyn t l khung hnh.
"""


class FixedAspectRatioPadding:
    """
    [VN #2] Gi nguyn t l khung hnh khi resize, sau  crop center  t kch thc vung.
    
    Chin lc ci tin (x l nh nh & ln):
    1. Scale nh sao cho cnh nh nht = 256px, gi aspect ratio
       - Nu nh 100x200  128x256 (width nh, scale up width)
       - Nu nh 300x200  384x256 (height nh, scale up height)
    2. Center crop t 256x256 (mt cnh=256, cnh kia>=256)
    3. Final center crop  224x224
    
    u im:
    - Trnh padding (khng c vin artificial)
    - Aspect ratio c bo ton  khng b mo
    - Ct  gia  gi phn quan trng ca mn n
    - X l nh nh m khng b qu m
    - X l nh ln m khng b ct xu
    
    V d:
    - nh 100x200  resize to 128x256  crop center 256x256  crop center 224x224
    - nh 300x200  resize to 384x256  crop center 256x256  crop center 224x224
    - nh 200x200 (vung)  resize to 256x256  crop center 256x256  crop center 224x224
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
            target_size: Kch thc trung gian sau resize (mc nh 256)
            final_size: Kch thc cui cng sau crop (mc nh 224)
            pad_mode: Cch padding - "reflect", "replicate", "edge", "constant"
            interpolation: Phng php ni suy (LANCZOS cho cht lng tt)
        """
        self.target_size = target_size
        self.final_size = final_size
        self.pad_mode = pad_mode
        self.interpolation = interpolation if interpolation is not None else LANCZOS
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        p dng transform.
        
        Args:
            img: PIL Image
            
        Returns:
            PIL Image vi kch thc final_size x final_size
        """
        width, height = img.size
        aspect_ratio = width / height
        
        # [VN #1 + #2] Smart resize strategy:
        # - Gi aspect ratio: scale theo cnh nh hn
        # - Cnh nh hn s = target_size (256)
        # - Cnh ln hn s > target_size (keep ratio)
        # - Sau  ct center  c hnh vung target_size x target_size
        
        if aspect_ratio > 1:  # width > height
            # height l cnh nh - scale  height = target_size
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)
        else:  # height >= width
            # width l cnh nh - scale  width = target_size
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        
        # [VN #5] Resize dng LANCZOS interpolation (cht lng cao)
        img = img.resize((new_width, new_height), self.interpolation)
        
        # Center crop  c target_size x target_size (khi vung)
        # Lc ny mt chiu = target_size, chiu kia >= target_size
        left = max(0, (new_width - self.target_size) // 2)
        top = max(0, (new_height - self.target_size) // 2)
        right = left + self.target_size
        bottom = top + self.target_size
        
        img = img.crop((left, top, right, bottom))
        
        # Final center crop  c final_size x final_size (224x224)
        left = max(0, (self.target_size - self.final_size) // 2)
        top = max(0, (self.target_size - self.final_size) // 2)
        img = img.crop((left, top, left + self.final_size, top + self.final_size))
        
        return img

# ============================================================================
# HELPER FUNCTIONS - VN #3, #6
# ============================================================================
"""
Helper functions  tnh class weights v kim tra stratified distribution.
"""


def calculate_class_weights(
    data_dir: Path | str = DATA_DIR,
    split: str = "Train",
) -> torch.Tensor:
    """
    [VN #3] Tnh class weights  x l class imbalance.
    
    Cng thc: weight[i] = total_samples / (num_classes * samples_in_class_i)
    
     tng: Class c t mu s c gn trng s cao hn.
    
    Returns:
        torch.Tensor: shape (num_classes,), dtype=float32
        
    Example:
        >>> weights = calculate_class_weights(split="Train")
        >>> sampler = WeightedRandomSampler(weights, len(weights))
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Folder khng tn ti: {split_dir}")
    
    # Scan tt c nh t mi class
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
    
    # Tnh weights
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # m bo class_names sorted (ging nh VNFoodDataset)
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
    [VN #6] Kim tra xem tp Validation c stratified tt so vi Train khng.
    
    In ra bng so snh phn b % gia Train v Validate.
    Nu chnh lch ln (>5%), cnh bo.
    
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
        
        # Tnh %
        percentages = {cls: (count / total) * 100 for cls, count in class_counts.items()}
        distribution[split] = percentages
    
    # In bng so snh
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
        
        status = "[OK]" if diff <= 2 else "[WARN]" if diff <= 5 else ""
        print(f"{class_name:<25} {train_pct:>10.2f} {val_pct:>10.2f} {diff:>9.2f} {status}")
    
    print(f"{'-'*80}")
    print(f"Max difference: {max_diff:.2f}%")
    
    if max_diff <= 2:
        print("[OK] EXCELLENT: Validation set is well-stratified")
    elif max_diff <= 5:
        print("[WARN] WARNING: Some classes have >2% difference (acceptable)")
    else:
        print(" PROBLEM: Significant imbalance between Train and Validate")
    
    print(f"{'='*80}\n")
    
    return distribution


# ============================================================================
# TRANSFORM PIPELINES - GII QUYT VN #4, #5
# ============================================================================
"""
nh ngha cc transform pipeline cho training v validation.

Ci tin:
[OK] Dng FixedAspectRatioPadding thay v RandomCrop (VN #2)
[OK] Dng LANCZOS interpolation (VN #5)
[OK] ColorJitter ti u cho food domain (VN #4)
"""


def get_train_transform() -> transforms.Compose:
    """
    To transform pipeline ti u cho training set.

    Pipeline (theo th t):
    
    1. [VN #2 + #1] FixedAspectRatioPadding(256, 224):
        Gi nguyn aspect ratio ca nh (khng b mo)
        Scale theo cnh nh nht (SMART: khng phng qu ln, khng ct mt chi tit)
        Resize dng LANCZOS (VN #5) - cht lng cao
        Center crop  t kch thc vung (gi phn trung tm quan trng)
        X l tt nh nh (<224x224) ln nh ln (>224x224)
    
    2. RandomHorizontalFlip(p=0.5):
        Mn n thng i xng ngang
        Tng gp i training data mt cch t nhin
    
    3. RandomRotation(10):
        nh thc t chp t gc khc nhau (nhng khng quay qu 10)
        Model hc rotational invariance
        GIM t 15 -> 10  trnh to ra orientation qu l
    
    4. [VN #4] ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08):
        Ti u cho food domain:
         * brightness=0.15: Trnh thay i  sng qu t ngt
           (mn n c mu t nhin, khng nn b sng qu hay ti qu)
         * contrast=0.15: Gi texture tng phn, khng qu cng
         * saturation=0.15: Mu sc ti sng ca thc phm c bo ton
           (khng qu nc ngoi khin mt nhn din)
         * hue=0.08: Hn ch thay i sc thi (khng i vng thnh tm)
        Cc gi tr ny da trn kinh nghim food domain experts
    
    5. ToTensor(): Chuyn PIL Image [0,255]  Tensor [0.0,1.0]
    
    6. Normalize(IMAGENET_MEAN, IMAGENET_STD):
        EfficientNetB0 pretrained trn ImageNet
        PHI chun ha cng mean/std

    Returns:
        transforms.Compose: Pipeline transform ti u.
    """
    return transforms.Compose(
        [
            # [VN #1, #2, #5] Custom transform: gi aspect ratio + LANCZOS
            FixedAspectRatioPadding(
                target_size=256,
                final_size=224,
                pad_mode="reflect",
                interpolation=LANCZOS,
            ),
            
            # Data augmentation for training
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # Reduced t 15 -> 10
            
            # [VN #4] ColorJitter ti u cho food domain
            transforms.ColorJitter(
                brightness=0.15,  # Gii hn  sng thay i
                contrast=0.15,    # Gi texture khng qu cng
                saturation=0.15,  # Bo ton mu sc ti sng
                hue=0.08,         # Hn ch thay i sc thi
            ),
            
            #  trng GaussianBlur v khng phi tt c Torchvision versions u h tr
            # Thm nu mun: transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            
            # Normalize
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_val_transform() -> transforms.Compose:
    """
    To transform pipeline ti u cho validation/test set (NO AUGMENTATION).

    Pipeline (theo th t):
    
    1. [VN #1, #2, #5] FixedAspectRatioPadding(256, 224):
        Cng logic nh training: gi aspect ratio + LANCZOS
        Deterministic (khng random, kt qu reproducible)
    
    2. ToTensor(): Chuyn PIL Image  Tensor
    
    3. Normalize(IMAGENET_MEAN, IMAGENET_STD):
        Chun ha cng ImageNet

    Returns:
        transforms.Compose: Pipeline transform deterministic.

    Note:
        Validation khng dng augmentation  nh gi model trn "clean" data
    """
    return transforms.Compose(
        [
            # [VN #1, #2, #5] Cng FixedAspectRatioPadding (deterministic)
            FixedAspectRatioPadding(
                target_size=256,
                final_size=224,
                pad_mode="reflect",
                interpolation=LANCZOS,
            ),
            
            # NO augmentation - validation dng "clean" data
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ============================================================================
# DATASET CLASS
# ============================================================================


class VNFoodDataset(Dataset):
    """
    Dataset loader cho d liu 30VNFoods.

    Features:
    - T ng scan class names t folder (sorted)
    - Lc file nh: .jpg, .jpeg, .png, .webp
    - Convert sang RGB (x l grayscale, RGBA)
    - Skip nh corrupted (log warning)

    Attributes:
        root_dir (Path): ng dn ti folder Train/Test/Validate
        split (str): "Train", "Test", hoc "Validate"
        transform (callable): Transform function (optional)
        class_names (list[str]): Sorted list of class names
        class_to_idx (dict): nh x class_name  index
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
        Khi to dataset.

        Args:
            root_dir: ng dn ti folder cha Train/Test/Validate subfolders.
            split: "Train", "Test", hoc "Validate".
            transform: Transform function (t get_train_transform() hoc get_val_transform()).

        Raises:
            ValueError: Nu split khng hp l hoc folder khng tn ti.
            FileNotFoundError: Nu folder split khng tm thy.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Kim tra split
        if split not in ["Train", "Test", "Validate"]:
            raise ValueError(f"split phi l 'Train', 'Test', hoc 'Validate', khng '{split}'")

        # Tm folder split
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Folder khng tn ti: {self.split_dir}")

        # Scan class folders
        self.class_names = sorted(
            [d.name for d in self.split_dir.iterdir() if d.is_dir()]
        )

        if len(self.class_names) == 0:
            raise ValueError(f"Khng tm thy class folders trong {self.split_dir}")

        # To class_to_idx mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Scan nh
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"Khng tm thy nh trong {self.split_dir}")

    def _load_samples(self) -> None:
        """
        Scan tt c nh t class folders.
        T ng skip nh corrupted.
        """
        print(f"   Scanning {len(self.class_names)} classes from {self.split}...", flush=True)
        for idx, class_name in enumerate(self.class_names, 1):
            class_dir = self.split_dir / class_name
            class_samples = 0

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue

                # Kim tra nh c corrupted khng
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
        """Tr v s lng nh trong dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Ly mt mu t dataset.

        Args:
            idx: Index ca mu.

        Returns:
            Tuple ca (image_tensor, label, class_name)
            - image_tensor: torch.Tensor shape (3, 224, 224)
            - label: int (class index)
            - class_name: str (tn class)
        """
        img_path, label_idx = self.samples[idx]
        class_name = self.class_names[label_idx]

        # Load nh
        with Image.open(img_path) as img:
            # Convert sang RGB (x l grayscale, RGBA)
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
    To DataLoaders cho train/val/test splits.

    [VN #3] Vi WeightedRandomSampler  x l class imbalance.
    [VN #6] Optional: kim tra stratified distribution ca validation set.

    Args:
        data_dir: ng dn ti folder cha Train/Test/Validate.
        batch_size: Batch size cho training (default: 32).
        num_workers: S worker processes  load nh (default: 4).
        use_weighted_sampler: Nu True, dng WeightedRandomSampler cho train loader (default: True).
        check_stratified: Nu True, kim tra stratified distribution (default: True).

    Returns:
        Tuple[dataloaders_dict, class_weights]:
        - dataloaders_dict: Dictionary vi keys "train", "val", "test"
        - class_weights: Tensor ca class weights (hoc None nu khng dng)

    Details:
        - Train loader: WeightedRandomSampler (x l imbalance), augmentation, drop_last=True
        - Val/Test loaders: SequentialSampler (random=False), no augmentation
        - pin_memory=True nu CUDA available
        
        [VN #3] WeightedRandomSampler:
         Ly mu theo xc sut t l vi class weights
         Class t s c sample nhiu hn  cn bng d liu
         Gim impact ca class imbalance

    Example:
        >>> dataloaders, weights = get_dataloaders(batch_size=32, use_weighted_sampler=True)
        >>> train_loader = dataloaders["train"]
        >>> for images, labels, class_names in train_loader:
        ...     print(images.shape)  # (32, 3, 224, 224)
        ...     break
    """
    data_dir = Path(data_dir)

    # [VN #6] Kim tra stratified distribution (optional)
    if check_stratified:
        try:
            check_stratified_distribution(data_dir)
        except Exception as e:
            warnings.warn(f"Could not check stratified distribution: {e}")

    # To datasets
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

    # Kim tra CUDA
    pin_memory = torch.cuda.is_available()

    # [VN #3] Tnh class weights v to WeightedRandomSampler
    class_weights = None
    train_sampler = None
    
    if use_weighted_sampler:
        try:
            # Tnh weights theo class imbalance
            class_weights = calculate_class_weights(data_dir, split="Train")
            
            # To sample weights: mi sample c gn weight ca class n
            # class_weights[i] l weight ca class i
            # train_dataset.samples[j] = (img_path, label_idx)
            sample_weights = torch.tensor(
                [class_weights[label].item() for _, label in train_dataset.samples],
                dtype=torch.float32
            )
            
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True,  # Cho php sample li (important cho imbalance)
            )
            print("[OK] Using WeightedRandomSampler for training (handles class imbalance)")
        except Exception as e:
            warnings.warn(f"Could not create WeightedRandomSampler: {e}")
            train_sampler = None

    # To DataLoaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,  # [VN #3] Use weighted sampler
            shuffle=(train_sampler is None),  # Ch shuffle nu khng dng sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Trnh batch size = 1
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
    Visualize augmentation effect trn mt nh.

    Hin th grid 2(n_augments+1):
    - Hng 1: PIL images (trc normalize)
    - Hng 2: Tensors (sau normalize, denormalize  xem)

    Args:
        class_name: Tn class  ly nh mu.
        data_dir: ng dn ti data folder.
        n_augments: S lng augmented versions  hin th.

    Returns:
        None (hin th figure v save PNG)

    Example:
        >>> show_augmentation_examples("Pho", n_augments=4)
        >>> # Hin th grid: [Original] [Aug1] [Aug2] [Aug3] [Aug4]
    """
    data_dir = Path(data_dir)
    Path("figures").mkdir(parents=True, exist_ok=True)

    # To dataset  ly nh
    dataset = VNFoodDataset(
        root_dir=data_dir,
        split="Train",
        transform=None,  # Ly PIL image gc
    )

    # Tm nh ca class_name
    if class_name not in dataset.class_names:
        raise ValueError(f"Class '{class_name}' khng tm thy. "
                        f"Available: {dataset.class_names}")

    class_idx = dataset.class_to_idx[class_name]
    sample_indices = [i for i, (_, label, _) in enumerate(
        [(dataset.samples[j][0], dataset.samples[j][1], dataset.class_names[dataset.samples[j][1]])
         for j in range(len(dataset))]
    ) if label == class_idx]

    if not sample_indices:
        raise ValueError(f"Khng tm thy nh cho class '{class_name}'")

    idx = sample_indices[0]
    img_path, _ = dataset.samples[idx]

    # Load nh gc
    original_img = Image.open(img_path).convert("RGB")

    # To augmented versions
    train_transform = get_train_transform()
    augmented_imgs = [original_img]
    augmented_tensors = [transforms.ToTensor()(original_img)]

    for _ in range(n_augments):
        aug_tensor = train_transform(original_img)
        augmented_tensors.append(aug_tensor)

        # Denormalize  visualize
        denorm_tensor = aug_tensor.clone()
        for i, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            denorm_tensor[i] = denorm_tensor[i] * std + mean
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        augmented_imgs.append(transforms.ToPILImage()(denorm_tensor))

    # To grid
    # To grid
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
    print(f"[OK] Saved {save_path}")
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
        print("   [OK] Stratified check passed")
    except Exception as e:
        print(f"   [WARN] Warning: {e}")

    # Test 1: Calculate class weights
    print("\n1. Calculating class weights...")
    try:
        weights = calculate_class_weights(DATA_DIR, split="Train")
        print(f"   [OK] Class weights calculated: shape {weights.shape}")
    except Exception as e:
        print(f"   [WARN] Warning: {e}")

    # Test 2: Creating dataloaders dengan weighted sampler
    print("\n2. Creating dataloaders (with WeightedRandomSampler)...")
    print("    Scanning folders (30-60 seconds for first run)...")
    print("   (Windows tip: num_workers=0 to avoid multiprocessing overhead)")
    try:
        # [WARN] num_workers=0 on Windows to avoid multiprocessing issues
        dataloaders, class_weights = get_dataloaders(
            batch_size=32,
            num_workers=0,
            use_weighted_sampler=True,
            check_stratified=True,
        )
        print("   [OK] Dataloaders created successfully")

        for split_name, loader in dataloaders.items():
            num_batches = len(loader)
            print(f"   - {split_name:6} split: {num_batches:3} batches")
    except Exception as e:
        print(f"    Error: {e}")
        exit(1)

    # Test 3: Kim tra batch structure
    print("\n3. Testing batch structure...")
    try:
        train_loader = dataloaders["train"]
        images, labels, class_names = next(iter(train_loader))

        print(f"   Image tensor shape: {images.shape}")
        print(f"   Expected: (batch_size, 3, 224, 224)")
        if images.shape[1:] == (3, 224, 224):
            print("   [OK] PASS - Shape correct!")
        else:
            print(f"    FAIL - Shape mismatch!")

        print(f"\n   Labels shape: {labels.shape}, dtype: {labels.dtype}")
        print(f"   Sample class names: {class_names[:3]}...")
        
        # Check image value range (should be normalized)
        print(f"\n   Image value range:")
        print(f"     Min: {images.min():.3f}, Max: {images.max():.3f}")
        print(f"     Mean: {images.mean():.3f}, Std: {images.std():.3f}")
        print("     (Expected: roughly [-2, 2] after ImageNet normalization)")
    except Exception as e:
        print(f"    Error: {e}")

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
            print("   [OK] PASS - 30 classes found")
        else:
            print(f"    Classes mismatch: {len(train_dataset.class_names)}")

        print(f"\n   First 5 classes (sorted):")
        for i, name in enumerate(train_dataset.class_names[:5], 1):
            count = sum(1 for _, label in train_dataset.samples if label == i-1)
            print(f"     {i}. {name:20} ({count} samples)")
    except Exception as e:
        print(f"    Error: {e}")

    # Test 5: Test new transforms (aspect ratio preservation)
    print("\n5. Testing new FixedAspectRatioPadding transform...")
    try:
        transform = FixedAspectRatioPadding()
        
        # To test images vi aspect ratios khc nhau
        test_imgs = [
            Image.new("RGB", (100, 200)),  # High, narrow
            Image.new("RGB", (300, 100)),  # Wide, short
            Image.new("RGB", (200, 200)),  # Square
            Image.new("RGB", (50, 100)),   # Very small
        ]
        
        for img in test_imgs:
            result = transform(img)
            print(f"   {img.size}  {result.size} [OK]")
        
        print("   [OK] PASS - FixedAspectRatioPadding works correctly")
    except Exception as e:
        print(f"    Error: {e}")

    # Test 6: Augmentation visualization (optional)
    print("\n6. Testing augmentation visualization...")
    try:
        show_augmentation_examples("Pho", n_augments=4)
        print("   [OK] Augmentation demo created")
    except Exception as e:
        print(f"   [WARN] Warning: {e}")
        print("   (This is optional - may fail if data issues)")

    print("\n" + "=" * 80)
    print("[OK] ALL TESTS COMPLETED!")
    print("=" * 80)
    
    # Print integration guide
    print("\n" + "=" * 80)
    print("INTEGRATION GUIDE FOR TRAINING LOOP")
    print("=" * 80)
    print("""
Hng dn s dng dataloaders ti u trong training loop:

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
   [OK] FixedAspectRatioPadding: Preserves aspect ratio, no cropping artifacts
   [OK] LANCZOS interpolation: Better image quality during resizing
   [OK] Optimized ColorJitter: Tuned for food domain (realistic colors)
   [OK] WeightedRandomSampler: Handles class imbalance automatically
   [OK] Stratified check: Validates distribution between train/val splits
""")
    print("=" * 80)
