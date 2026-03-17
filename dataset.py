"""
dataset.py — Custom Dataset cho pipeline DeepLabV3+.

FoodSemSeg format:
  - ảnh: train/images/*.jpg
  - mask: train/masks/*.png (pixel-level semantic, sparse category IDs)
  - category_info.json: mapping sparse ID -> sequential

Tính năng:
- skip_resize: bỏ A.Resize nếu data đã được resize offline (đọc thẳng từ đĩa, nhanh hơn).
- Aug nhẹ (Flip, Brightness). Cấm Elastic / Grid.
- mask: integer label sequential 0..num_classes-1 (background=0).
"""

import os
import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger("DeepLabV3_FineTune")


# ============================================================
# FOODSEMSEG DATASET — PNG masks, sparse category IDs
# ============================================================
class FoodSemSegDataset(Dataset):
    """
    Dataset cho FoodSemSeg: ảnh + mask PNG.

    Args:
        skip_resize: True nếu data đã resize offline → bỏ A.Resize (tiết kiệm CPU).
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        num_classes: int,
        raw_to_seq: dict,
        split: str = "train",
        img_size: tuple = (512, 512),
        skip_resize: bool = False,
        rare_class_ids: list = None,
        strong_rare_aug: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.raw_to_seq = raw_to_seq
        self.split = split
        self.img_size = img_size  # (W, H)
        self.rare_class_ids = set(rare_class_ids or [])
        self.strong_rare_aug = bool(strong_rare_aug and self.rare_class_ids)

        self.images = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        # ---- Build transform pipeline ----
        transforms = []

        # Resize chỉ khi CHƯA resize offline
        if not skip_resize:
            transforms.append(A.Resize(height=img_size[1], width=img_size[0]))

        # Aug chỉ cho train
        if split == "train":
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.Affine(
                    scale=(0.85, 1.15),
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-15, 15),
                    p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20,
                    val_shift_limit=15, p=0.3,
                ),
                A.GaussNoise(p=0.2),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    fill=0, fill_mask=0, p=0.3,
                ),
            ])
            # Augmentation mạnh hơn riêng cho ảnh chứa class hiếm (khi bật strong_rare_aug)
            if self.strong_rare_aug:
                self.transform_strong_rare = A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=25, p=0.5),
                    A.GaussNoise(var_limit=(10, 50), p=0.3),
                ])
            else:
                self.transform_strong_rare = None
        else:
            self.transform_strong_rare = None

        # Normalize + ToTensor luôn có
        transforms.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.images)

    def _raw_to_sequential(self, mask: np.ndarray) -> np.ndarray:
        """Map sparse category IDs -> sequential 0..num_classes-1."""
        if self.raw_to_seq is None:
            return np.clip(mask.astype(np.int64), 0, self.num_classes - 1)
        out = np.zeros_like(mask, dtype=np.int64)
        for raw_id, seq_id in self.raw_to_seq.items():
            out[mask == raw_id] = seq_id
        return out

    def _load_from_disk(self, img_name: str):
        """Đọc ảnh + mask từ disk, chuyển RGB, map class IDs."""
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Không đọc được mask: {mask_path}")
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        mask = self._raw_to_sequential(mask.astype(np.int64))
        return image, mask

    def __getitem__(self, idx):
        # Đọc trực tiếp từ ổ disk
        image, mask = self._load_from_disk(self.images[idx])

        # Nếu ảnh chứa class hiếm và bật strong_rare_aug: áp dụng aug mạnh trước
        if self.transform_strong_rare is not None and self.rare_class_ids:
            has_rare = np.any(np.isin(mask, list(self.rare_class_ids)))
            if has_rare:
                aug_strong = self.transform_strong_rare(image=image, mask=mask)
                image, mask = aug_strong["image"], aug_strong["mask"]

        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].long()


# Alias để tương thích với train.py (FoodSegDataset -> FoodSemSegDataset)
FoodSegDataset = FoodSemSegDataset


# ============================================================
# CLASS WEIGHTS — tính distribution từ mask PNG
# ============================================================
def calculate_class_weights(
    label_dir: str,
    image_dir: str,
    num_classes: int,
    raw_to_seq: dict = None,
) -> torch.Tensor:
    """
    Quét mask PNG → đếm pixel per class.
    Trả về class_weights (dùng cho CE Loss).
    raw_to_seq: mapping sparse ID -> sequential. Nếu None, giả sử mask đã sequential.
    """
    logger.info("Tính Class Distribution từ mask files...")
    counts = np.zeros(num_classes, dtype=np.int64)

    images = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for img_name in images:
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(label_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        if raw_to_seq is not None:
            out = np.zeros_like(mask, dtype=np.int64)
            for raw_id, seq_id in raw_to_seq.items():
                out[mask == raw_id] = seq_id
            mask = out
        else:
            mask = mask.astype(np.int64)

        mask = np.clip(mask, 0, num_classes - 1)
        unique, cnt = np.unique(mask, return_counts=True)
        for u, c in zip(unique, cnt):
            if 0 <= u < num_classes:
                counts[u] += c

    total = counts.sum()
    # Median frequency balancing — mạnh hơn 1/log cho class imbalance cực
    freqs = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if counts[c] > 0:
            freqs[c] = counts[c] / total

    nonzero_freqs = freqs[freqs > 0]
    median_freq = np.median(nonzero_freqs) if len(nonzero_freqs) > 0 else 1.0

    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = median_freq / freqs[c]
        else:
            weights[c] = 0.0

    # Clip để tránh extreme weights (cap 5.0 thay 10.0 cho ổn định hơn)
    weights = np.clip(weights, 0.1, 5.0)

    logger.info(f"Class pixel counts : {counts}")
    logger.info(f"Class weights (CE) : {np.round(weights, 3)}")
    return torch.tensor(weights, dtype=torch.float32)


def build_rare_oversampler(
    label_dir: str,
    image_dir: str,
    num_classes: int,
    rare_classes: list,
    alpha: float = 2.0,
):
    """
    Oversample ảnh có chứa rare_classes. Trả về (sampler, weights_per_image).
    Heuristic: weight = 1 + alpha * (#rare_present).
    """
    images = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    weights = np.ones(len(images), dtype=np.float64)
    rare_set = set(int(c) for c in rare_classes)

    for i, img_name in enumerate(images):
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(label_dir, mask_name)
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim > 2:
            m = m[:, :, 0]
        m = np.clip(m.astype(np.int64), 0, num_classes - 1)
        uniq = np.unique(m)
        present = sum((int(u) in rare_set) for u in uniq)
        if present > 0:
            weights[i] = 1.0 + alpha * present

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, weights
