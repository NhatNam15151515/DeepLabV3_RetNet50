"""
Visualize toàn bộ data train + test:
Mỗi ảnh: [image | mask màu | legend class có trong ảnh].
Output: visualize_train/, visualize_test/ tại project root.

Cách dùng:
    python scripts/data_tools/visualize_all_data.py
"""

import os
import numpy as np
import cv2

import config as CFG

NUM_CLASSES = CFG.NUM_CLASSES
CLASS_NAMES = getattr(CFG, "CLASS_NAMES", [f"class_{i}" for i in range(NUM_CLASSES)])
TRAIN_IMG_DIR = CFG.TRAIN_IMG_DIR
TRAIN_LABEL_DIR = CFG.TRAIN_LABEL_DIR
VAL_IMG_DIR = CFG.VAL_IMG_DIR
VAL_LABEL_DIR = CFG.VAL_LABEL_DIR
OUT_TRAIN = os.path.join(CFG.PROJECT_ROOT, "visualize_train")
OUT_TEST = os.path.join(CFG.PROJECT_ROOT, "visualize_test")

_PALETTE = [
    (0, 0, 0),       # 0 background - den
    (128, 0, 0),     # 1 egg
    (0, 128, 0),     # 2 banana
    (128, 128, 0),   # 3 steak
    (0, 0, 128),     # 4 pork
    (128, 0, 128),   # 5 chicken duck
    (0, 128, 128),   # 6 fish
    (128, 128, 128), # 7 shrimp
    (64, 0, 0),      # 8 bread
    (192, 0, 0),     # 9 noodles
    (64, 128, 0),    # 10 rice
    (192, 128, 0),   # 11 tofu
    (64, 0, 128),    # 12 potato
    (192, 0, 128),   # 13 tomato
    (64, 128, 128),  # 14 lettuce
    (192, 128, 128), # 15 cucumber
    (0, 64, 0),      # 16 carrot
    (128, 64, 0),    # 17 broccoli
    (0, 192, 0),     # 18 cabbage
    (128, 192, 0),   # 19 onion
    (0, 64, 128),    # 20 pepper
    (128, 64, 128),  # 21 French beans / other
]


def _colored_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(NUM_CLASSES, len(_PALETTE))):
        out[mask == c] = _PALETTE[c]
    return out


def _draw_legend(img_h: int, present_ids: list) -> np.ndarray:
    ids = [c for c in present_ids if 0 <= c < NUM_CLASSES]
    if not ids:
        ids = [0]
    line_h = 22
    pad = 8
    n = len(ids)
    legend_h = n * line_h + 2 * pad
    legend_w = 200
    legend = np.ones((max(img_h, legend_h), legend_w, 3), dtype=np.uint8) * 255
    for idx, cid in enumerate(ids):
        y = pad + idx * line_h + line_h // 2
        color = _PALETTE[cid] if cid < len(_PALETTE) else (128, 128, 128)
        cv2.rectangle(legend, (pad, pad + idx * line_h), (pad + 16, pad + (idx + 1) * line_h - 2), color, -1)
        cv2.rectangle(legend, (pad, pad + idx * line_h), (pad + 16, pad + (idx + 1) * line_h - 2), (0, 0, 0), 1)
        name = CLASS_NAMES[cid][:20] if cid < len(CLASS_NAMES) else f"class_{cid}"
        label = f"{cid}: {name}"
        cv2.putText(legend, label, (pad + 20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return legend[:img_h, :]


def process_split(image_dir: str, label_dir: str, out_root: str, split_name: str) -> None:
    os.makedirs(out_root, exist_ok=True)
    images = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + ".png"
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(label_dir, mask_name)
        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            continue
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            continue
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        mask = np.clip(mask.astype(np.int32), 0, NUM_CLASSES - 1)
        present_ids = sorted(np.unique(mask).astype(int).tolist())
        h, w = image.shape[:2]
        mask_colored = _colored_mask(mask)
        legend = _draw_legend(h, present_ids)
        row = np.hstack([image, mask_colored, legend])
        cv2.putText(row, "image", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(row, "mask (labels)", (w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(row, "labels in image", (2 * w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        out_path = os.path.join(out_root, f"{base_name}.png")
        cv2.imwrite(out_path, row)
    print(f"[{split_name}] {len(images)} ảnh -> {out_root}")


def main():
    print("Visualize all train + test with mask labels (legend)...")
    process_split(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, OUT_TRAIN, "train")
    process_split(VAL_IMG_DIR, VAL_LABEL_DIR, OUT_TEST, "test")
    print("Done. visualize_train/, visualize_test/")


if __name__ == "__main__":
    main()

