"""
test_nv_subset_pipeline.py

Test pipeline 3 tầng (segmentation → depth → volume → weight) trên subset NutritionVerse-real.

Phiên bản mới này:
- Dùng cùng model + preprocess với scripts/api/api.py (không bóp méo ảnh gốc).
- Tier 1: chạy segmentation model trên ảnh gốc, union toàn bộ vùng foreground (class > 0) làm mask.
- Tier 2: dùng MiDaS (Tier2DepthVolume) để ước lượng depth + thể tích từ mask thực tế.
- Tier 3: dùng Tier3WeightEstimation để ước lượng trọng lượng (density theo NutritionVerse).

Kết quả:
- CSV: nutritionVerse-real/subset_100_filtered/pipeline_results.csv
  với các cột: file_name, dish_id, class_name, gt_weight, pred_weight, abs_error, rel_error, volume, density.
- Visual: nutritionVerse-real/subset_100_filtered/pipeline_vis/
  mỗi ảnh có: *_overlay.png (ảnh overlay có chú thích legend).

Chạy:
    python scripts/nv_pipeline/test_nv_subset_pipeline.py
"""

import csv
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import config as CFG
from model_setup import create_model
from nv_pipeline.tier2_depth_volume import Tier2DepthVolume, DepthVolumeResult
from nv_pipeline.tier3_weight_estimation import Tier3WeightEstimation, WeightEstimationResult

ROOT = Path(r"c:\Nhat Nam\do an chuyen nganh\DeepLabV3_RetNet50")
SUBSET_CSV = ROOT / "nutritionVerse-real" / "subset_100_filtered" / "selection_100.csv"

# Anh chỉnh lại root này đúng chỗ để script tìm được ảnh
NV_IMAGE_ROOT = ROOT / "nutritionVerse-real" / "nutritionverse-manual" / "nutritionverse-manual" / "images"

OUT_CSV = ROOT / "nutritionVerse-real" / "subset_100_filtered" / "pipeline_results.csv"
OUT_VIS_DIR = ROOT / "nutritionVerse-real" / "subset_100_filtered" / "pipeline_vis"

OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)

# ===== Segmentation config giống api.py =====
API_BEST_PATH = os.path.join(
    CFG.RUNS_DIR, "exp21", "weights", "best.pth"
)
MAX_INPUT_SIDE = 512  # cạnh dài đưa vào model

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Palette 22 class (giống scripts/api/api.py)
CLASS_COLORS = [
    (0, 0, 0),  # 0  background
    (255, 255, 255),  # 1  egg
    (255, 215, 0),  # 2  banana
    (205, 92, 92),  # 3  steak
    (210, 105, 30),  # 4  pork
    (255, 165, 0),  # 5  chicken
    (70, 130, 180),  # 6  fish
    (255, 140, 0),  # 7  shrimp
    (222, 184, 135),  # 8  bread
    (255, 228, 196),  # 9  noodles
    (238, 232, 170),  # 10 rice
    (245, 245, 220),  # 11 tofu
    (210, 180, 140),  # 12 potato
    (255, 99, 71),  # 13 tomato
    (144, 238, 144),  # 14 lettuce
    (50, 205, 50),  # 15 cucumber
    (255, 165, 79),  # 16 carrot
    (34, 139, 34),  # 17 broccoli
    (154, 205, 50),  # 18 cabbage
    (255, 255, 0),  # 19 onion
    (255, 69, 0),  # 20 pepper
    (255, 182, 193),  # 21 other
]


def draw_mask_legend(img: np.ndarray, present_classes: list, class_names: list, class_colors: list) -> np.ndarray:
    """Vẽ chú thích legend (chỉ class có trong mask) lên góc ảnh, font to dễ đọc."""
    present_classes = [c for c in present_classes if c > 0]
    if not present_classes:
        return img
    out = img.copy()
    pad = 30
    row_h = 80
    patch_w = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 4
    n = len(present_classes)
    box_w = 700
    box_h = pad * 2 + n * row_h
    ih, iw = out.shape[:2]
    x0, y0 = iw - box_w, 0
    overlay_box = out.copy()
    cv2.rectangle(overlay_box, (x0, y0), (iw, box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay_box, 0.75, out, 0.25, 0, out)
    cv2.rectangle(out, (x0, y0), (iw, box_h), (180, 180, 180), 2)
    x, y = x0 + pad, pad
    for cid in sorted(present_classes):
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        color = tuple(int(c) for c in class_colors[cid]) if cid < len(class_colors) else (255, 255, 255)
        cv2.rectangle(out, (x, y + 2), (x + patch_w, y + row_h - 6), color, -1)
        cv2.rectangle(out, (x, y + 2), (x + patch_w, y + row_h - 6), (220, 220, 220), 2)
        cv2.putText(out, name, (x + patch_w + 8, y + row_h - 10), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += row_h
    return out


def preprocess_any_size(image: np.ndarray, max_side: int = MAX_INPUT_SIDE):
    """
    Tiền xử lý ảnh bất kỳ kích thước, không bóp méo (giống api.py):
    - Scale để cạnh dài = max_side (giữ tỷ lệ)
    - Pad về max_side x max_side (pad bottom-right, value=0)
    Trả về: (tensor 1xCxHxW, content_h, content_w).
    """
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, new_h, new_w


def preds_to_original_size(preds: np.ndarray, content_h: int, content_w: int, orig_h: int, orig_w: int):
    """Crop vùng nội dung (bỏ padding), resize về kích thước ảnh gốc."""
    cropped = preds[:content_h, :content_w].copy()
    out = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return out


class SegModelSingleton:
    """Singleton segmentation model giống API (DeepLabV3+ ResNet50)."""

    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(CFG.NUM_CLASSES, use_imagenet=True)
        state = torch.load(API_BEST_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        print(
            f"[Tier1] Segmentation model ready on {self.device} | "
            f"classes={CFG.NUM_CLASSES} | max_side={MAX_INPUT_SIDE}"
        )

    @classmethod
    def get(cls) -> "SegModelSingleton":
        if cls._instance is None:
            cls._instance = SegModelSingleton()
        return cls._instance


def run_segmentation(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chạy segmentation trên ảnh BGR gốc.
    Trả về:
        preds_orig: HxW (id class 0..C-1)
        fg_mask:    HxW (uint8, 1 = foreground union, 0 = background)
    """
    ih, iw = image.shape[:2]
    seg = SegModelSingleton.get()
    tensor, content_h, content_w = preprocess_any_size(image, MAX_INPUT_SIDE)
    tensor = tensor.to(seg.device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(seg.device.type == "cuda")):
        logits = seg.model(tensor)
        preds = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    preds_orig = preds_to_original_size(preds, content_h, content_w, ih, iw)
    fg_mask = (preds_orig > 0).astype(np.uint8)
    return preds_orig, fg_mask


def build_color_overlay(
    image: np.ndarray, preds_orig: np.ndarray, class_names: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo color_mask + overlay giống API, theo CLASS_COLORS, kích thước ảnh gốc.
    Thêm chú thích legend cho các class có trong mask.
    """
    ih, iw = image.shape[:2]
    color_mask = np.zeros((ih, iw, 3), dtype=np.uint8)
    for cid, color in enumerate(CLASS_COLORS):
        color_mask[preds_orig == cid] = color
    overlay = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)
    present_classes = sorted(set(int(x) for x in np.unique(preds_orig) if x > 0))
    if present_classes:
        color_mask = draw_mask_legend(color_mask, present_classes, class_names, CLASS_COLORS)
        overlay = draw_mask_legend(overlay, present_classes, class_names, CLASS_COLORS)
    return color_mask, overlay


def pick_primary_class(row: dict) -> str:
    """
    Chọn class chính cho dish (theo NutritionVerse metadata):
    - Lấy food_item_type_k có weight lớn nhất trong số các entry không rỗng.
    - Nếu không có, trả về 'unknown'.
    """
    best_type = "unknown"
    best_w = -1.0
    for i in range(1, 8):
        t_key = f"food_item_type_{i}"
        w_key = f"food_weight_g_{i}"
        t_val = (row.get(t_key) or "").strip()
        if not t_val:
            continue
        try:
            w_val = float(row.get(w_key) or 0.0)
        except ValueError:
            w_val = 0.0
        if w_val > best_w:
            best_w = w_val
            best_type = t_val
    return best_type or "unknown"


def main():
    print("[INFO] SUBSET_CSV   =", SUBSET_CSV)
    print("[INFO] NV_IMAGE_ROOT =", NV_IMAGE_ROOT)
    print("[INFO] OUT_CSV      =", OUT_CSV)
    print("[INFO] OUT_VIS_DIR  =", OUT_VIS_DIR)

    if not SUBSET_CSV.is_file():
        print("[ERROR] selection_100.csv không tồn tại. Hãy chạy select_nv_subset_100.py trước.")
        return

    # Khởi tạo Tier 2 & Tier 3 & Tier 1 seg
    tier2 = Tier2DepthVolume()
    tier3 = Tier3WeightEstimation()
    SegModelSingleton.get()  # load model 1 lần

    rows: List[dict] = []
    with open(SUBSET_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print("[INFO] Num rows from subset CSV:", len(rows))

    results: List[dict] = []

    for row in rows:
        fname = row["file_name"]
        dish_id = row["dish_id"]
        img_path = NV_IMAGE_ROOT / fname

        if not img_path.is_file():
            print("[WARN] Missing image:", img_path)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print("[WARN] Cannot read image:", img_path)
            continue

        # Tier 1: segmentation để lấy mask foreground (union tất cả class > 0)
        preds_orig, fg_mask = run_segmentation(img)

        # Tier 2: depth + volume, dùng mask thực tế
        depth_map = tier2.estimate_depth(img)
        dv: DepthVolumeResult = tier2.estimate_volume(
            img, [(pick_primary_class(row), fg_mask)], depth_map
        )[0]

        # Tier 3: weight
        we: WeightEstimationResult = tier3.estimate_weight(
            class_name=dv.class_name,
            volume_cm3=dv.volume_cm3,
        )

        try:
            gt_weight = float(row.get("total_food_weight") or 0.0)
        except ValueError:
            gt_weight = 0.0

        pred_weight = float(we.weight_grams)
        abs_err = abs(pred_weight - gt_weight)
        rel_err = abs_err / gt_weight if gt_weight > 1e-6 else 0.0

        results.append(
            {
                "file_name": fname,
                "dish_id": dish_id,
                "class_name": dv.class_name,
                "gt_weight_g": gt_weight,
                "pred_weight_g": pred_weight,
                "abs_error_g": abs_err,
                "rel_error": rel_err,
                "volume_cm3": dv.volume_cm3,
                "density_g_per_cm3": we.density_g_per_cm3,
                "density_source": we.density_source,
            }
        )

        # Lưu overlay có chú thích legend
        stem = Path(fname).stem
        class_names = getattr(CFG, "CLASS_NAMES", [f"class_{i}" for i in range(CFG.NUM_CLASSES)])
        _, overlay = build_color_overlay(img, preds_orig, class_names)
        cv2.imwrite(str(OUT_VIS_DIR / f"{stem}_overlay.png"), overlay)

    # Ghi CSV kết quả
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "dish_id",
        "class_name",
        "gt_weight_g",
        "pred_weight_g",
        "abs_error_g",
        "rel_error",
        "volume_cm3",
        "density_g_per_cm3",
        "density_source",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("[DONE] Saved pipeline results to:", OUT_CSV)
    print("[DONE] Num successful images:", len(results))


if __name__ == "__main__":
    main()
