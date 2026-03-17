"""
Thống kê lại các class trong FoodSemSeg_512x512 (sau khi merge thêm data):
- Đếm số ảnh chứa từng class (train/test riêng, và tổng).
- Đếm tổng số pixel từng class (thô, để tham khảo).

Cách dùng:
    python scripts/data_tools/compute_class_stats.py
"""

import os
import numpy as np
import cv2

import config as CFG


def collect_stats(img_dir: str, msk_dir: str, num_classes: int):
    images = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    images.sort()

    img_count_per_class = np.zeros(num_classes, dtype=np.int64)
    pixel_count_per_class = np.zeros(num_classes, dtype=np.int64)

    for img_name in images:
        base, _ = os.path.splitext(img_name)
        msk_name = base + ".png"
        msk_path = os.path.join(msk_dir, msk_name)
        if not os.path.isfile(msk_path):
            continue
        mask = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int32)
        mask = np.clip(mask, 0, num_classes - 1)

        unique, counts = np.unique(mask, return_counts=True)
        for cls, cnt in zip(unique, counts):
            cls = int(cls)
            if 0 <= cls < num_classes:
                pixel_count_per_class[cls] += int(cnt)
        for cls in unique:
            cls = int(cls)
            if 0 < cls < num_classes:
                img_count_per_class[cls] += 1

    return img_count_per_class, pixel_count_per_class


def main():
    num_classes = CFG.NUM_CLASSES
    class_names = getattr(CFG, "CLASS_NAMES", [f"class_{i}" for i in range(num_classes)])

    print("== Class stats for FoodSemSeg_512x512 ==")
    print("Num classes:", num_classes)
    print()

    train_img_counts, train_px_counts = collect_stats(CFG.TRAIN_IMG_DIR, CFG.TRAIN_LABEL_DIR, num_classes)
    test_img_counts, test_px_counts = collect_stats(CFG.VAL_IMG_DIR, CFG.VAL_LABEL_DIR, num_classes)

    total_img_counts = train_img_counts + test_img_counts
    total_px_counts = train_px_counts + test_px_counts

    header = f"{'id':>3} | {'class':<20} | {'train_img':>9} | {'test_img':>8} | {'total_img':>9} | {'px_total':>12}"
    print(header)
    print("-" * len(header))

    for cls in range(num_classes):
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(
            f"{cls:>3} | "
            f"{name:<20} | "
            f"{int(train_img_counts[cls]):>9} | "
            f"{int(test_img_counts[cls]):>8} | "
            f"{int(total_img_counts[cls]):>9} | "
            f"{int(total_px_counts[cls]):>12}"
        )


if __name__ == "__main__":
    main()

