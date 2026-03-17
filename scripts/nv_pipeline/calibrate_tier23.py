"""
calibrate_tier23.py

Phân tích + calibrate Tier2-3 dựa trên 38 ảnh NutritionVerse-real (sau loại ảnh sai).

Loại bỏ class: onion, rice, potato, noodles, pork (không dùng để calibrate).
Hai phương pháp calibrate:
  A) Global scale factor (1 hệ số cho tất cả)
  B) Per-class effective density (tối ưu riêng cho từng loại thức ăn NV)

Chạy:
    $env:PYTHONIOENCODING='utf-8'; python scripts/nv_pipeline/calibrate_tier23.py
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"c:\Nhat Nam\do an chuyen nganh\DeepLabV3_RetNet50")
RESULTS_CSV = ROOT / "nutritionVerse-real" / "subset_100_filtered" / "pipeline_results.csv"
SELECTION_CSV = ROOT / "nutritionVerse-real" / "subset_100_filtered" / "selection_100.csv"

EXCLUDE_CLASSES = {"onion", "rice", "potato", "noodles", "pork"}


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def get_per_item_gt(sel_rows):
    result = {}
    for row in sel_rows:
        fname = row["file_name"]
        items = []
        for i in range(1, 8):
            t = (row.get(f"food_item_type_{i}") or "").strip()
            w = row.get(f"food_weight_g_{i}") or ""
            if t:
                try:
                    w = float(w)
                except ValueError:
                    w = 0.0
                items.append((t, w))
        result[fname] = items
    return result


def main():
    results = load_csv(RESULTS_CSV)
    sel_rows = load_csv(SELECTION_CSV)
    per_item_gt = get_per_item_gt(sel_rows)

    # Filter: chỉ lấy ảnh KHÔNG có class bị loại
    valid = []
    for row in results:
        fname = row["file_name"]
        items_gt = per_item_gt.get(fname, [])
        item_types = set(t.lower() for t, _ in items_gt)
        if item_types & EXCLUDE_CLASSES:
            continue
        gt_w = float(row["gt_weight_g"])
        pred_w = float(row["pred_weight_g"])
        vol = float(row["volume_cm3"])
        density = float(row["density_g_per_cm3"])
        cls = row["class_name"].strip().lower()
        valid.append({
            "file_name": fname,
            "class_name": cls,
            "gt_weight_g": gt_w,
            "pred_weight_g": pred_w,
            "volume_cm3": vol,
            "density": density,
            "ratio_pred_gt": pred_w / gt_w if gt_w > 0 else 0,
        })

    print("=" * 90)
    print("PHAN TICH PIPELINE TIER2-3 TRUOC CALIBRATION")
    print(f"Tong anh hop le (khong co class bi loai): {len(valid)} / {len(results)}")
    print("=" * 90)

    # ========= TRUOC CALIBRATION =========
    by_class = defaultdict(list)
    for e in valid:
        by_class[e["class_name"]].append(e)

    print(f"\n{'CLASS':<25} {'N':>4} {'GT_AVG':>10} {'PRED_AVG':>12} {'RATIO':>10} {'VOL_AVG':>12}")
    print("-" * 80)
    for cls in sorted(by_class.keys()):
        entries = by_class[cls]
        n = len(entries)
        gt_avg = statistics.mean(e["gt_weight_g"] for e in entries)
        pred_avg = statistics.mean(e["pred_weight_g"] for e in entries)
        ratio_avg = statistics.mean(e["ratio_pred_gt"] for e in entries)
        vol_avg = statistics.mean(e["volume_cm3"] for e in entries)
        print(f"{cls:<25} {n:>4} {gt_avg:>10.1f} {pred_avg:>12.1f} {ratio_avg:>10.1f}x {vol_avg:>12.1f}")

    mae_before = statistics.mean(abs(e["pred_weight_g"] - e["gt_weight_g"]) for e in valid)
    mape_before = statistics.mean(
        abs(e["pred_weight_g"] - e["gt_weight_g"]) / e["gt_weight_g"]
        for e in valid if e["gt_weight_g"] > 0
    ) * 100

    print(f"\n--- TRUOC CALIBRATION ---")
    print(f"  MAE  = {mae_before:.1f} g")
    print(f"  MAPE = {mape_before:.1f} %")

    # ========= METHOD A: Global scale factor =========
    gt_pred_ratios = sorted(e["gt_weight_g"] / e["pred_weight_g"] for e in valid if e["pred_weight_g"] > 0)
    global_scale_median = statistics.median(gt_pred_ratios)
    global_scale_mean = statistics.mean(gt_pred_ratios)

    print(f"\n{'=' * 90}")
    print("METHOD A: Global scale factor")
    print(f"  Median = {global_scale_median:.6f}")
    print(f"  Mean   = {global_scale_mean:.6f}")

    mae_A = statistics.mean(abs(e["pred_weight_g"] * global_scale_median - e["gt_weight_g"]) for e in valid)
    mape_A = statistics.mean(
        abs(e["pred_weight_g"] * global_scale_median - e["gt_weight_g"]) / e["gt_weight_g"]
        for e in valid if e["gt_weight_g"] > 0
    ) * 100

    print(f"  MAE  after = {mae_A:.1f} g (truoc: {mae_before:.1f})")
    print(f"  MAPE after = {mape_A:.1f} % (truoc: {mape_before:.1f})")

    # ========= METHOD B: Per-class effective density =========
    # effective_density = gt / volume (absorbs volume error + real density)
    print(f"\n{'=' * 90}")
    print("METHOD B: Per-class effective density (gt_weight / volume)")
    print(f"{'CLASS':<25} {'N':>4} {'EFF_DENSITY_MEDIAN':>20} {'EFF_DENSITY_MEAN':>18}")
    print("-" * 72)

    class_density_median = {}
    class_density_mean = {}
    for cls in sorted(by_class.keys()):
        entries = by_class[cls]
        densities = [e["gt_weight_g"] / e["volume_cm3"] for e in entries if e["volume_cm3"] > 0]
        d_med = statistics.median(densities)
        d_mean = statistics.mean(densities)
        class_density_median[cls] = d_med
        class_density_mean[cls] = d_mean
        print(f"{cls:<25} {len(densities):>4} {d_med:>20.6f} {d_mean:>18.6f}")

    # Simulate Method B (dùng median density per class)
    calibrated_B = []
    for e in valid:
        d = class_density_median.get(e["class_name"], 0.9)
        new_pred = e["volume_cm3"] * d
        calibrated_B.append({
            **e,
            "new_pred": new_pred,
            "new_abs_err": abs(new_pred - e["gt_weight_g"]),
            "new_rel_err": abs(new_pred - e["gt_weight_g"]) / e["gt_weight_g"] if e["gt_weight_g"] > 0 else 0,
        })

    mae_B = statistics.mean(c["new_abs_err"] for c in calibrated_B)
    mape_B = statistics.mean(c["new_rel_err"] for c in calibrated_B) * 100

    print(f"\n  MAE  after = {mae_B:.1f} g (truoc: {mae_before:.1f})")
    print(f"  MAPE after = {mape_B:.1f} % (truoc: {mape_before:.1f})")

    # ========= METHOD C: Global vol_scale + real density =========
    # Giai doan 1: gan real density cho tung class
    # Giai doan 2: tinh vol_scale = median(gt / (vol * real_density))
    REAL_DENSITIES = {
        "chicken-wing": 1.05,
        "chicken-leg": 1.05,
        "chicken-breast": 1.04,
        "near-whole-chicken": 0.95,
        "lasagna": 1.10,
        "steak-piece": 1.08,
    }
    print(f"\n{'=' * 90}")
    print("METHOD C: Real density per-class + global volume scale")
    print("  Real densities used:")
    for k, v in REAL_DENSITIES.items():
        print(f"    {k}: {v}")

    vol_scale_ratios = []
    for e in valid:
        rd = REAL_DENSITIES.get(e["class_name"], 0.9)
        if e["volume_cm3"] > 0:
            ideal_vol = e["gt_weight_g"] / rd
            vol_scale_ratios.append(ideal_vol / e["volume_cm3"])

    vol_scale_median = statistics.median(vol_scale_ratios)
    vol_scale_mean = statistics.mean(vol_scale_ratios)
    print(f"  Volume scale (median) = {vol_scale_median:.6f}")
    print(f"  Volume scale (mean)   = {vol_scale_mean:.6f}")

    calibrated_C = []
    for e in valid:
        rd = REAL_DENSITIES.get(e["class_name"], 0.9)
        new_vol = e["volume_cm3"] * vol_scale_median
        new_pred = new_vol * rd
        calibrated_C.append({
            **e,
            "new_pred": new_pred,
            "new_abs_err": abs(new_pred - e["gt_weight_g"]),
            "new_rel_err": abs(new_pred - e["gt_weight_g"]) / e["gt_weight_g"] if e["gt_weight_g"] > 0 else 0,
        })

    mae_C = statistics.mean(c["new_abs_err"] for c in calibrated_C)
    mape_C = statistics.mean(c["new_rel_err"] for c in calibrated_C) * 100
    print(f"  MAE  after = {mae_C:.1f} g (truoc: {mae_before:.1f})")
    print(f"  MAPE after = {mape_C:.1f} % (truoc: {mape_before:.1f})")

    # ========= TONG KET =========
    print(f"\n{'=' * 90}")
    print("TONG KET SO SANH")
    print(f"{'=' * 90}")
    print(f"{'METHOD':<45} {'MAE(g)':>10} {'MAPE(%)':>10}")
    print("-" * 70)
    print(f"{'Truoc calibration (default density=0.9)':<45} {mae_before:>10.1f} {mape_before:>10.1f}")
    print(f"{'A) Global scale (median)':<45} {mae_A:>10.1f} {mape_A:>10.1f}")
    print(f"{'B) Per-class effective density (median)':<45} {mae_B:>10.1f} {mape_B:>10.1f}")
    print(f"{'C) Real density + vol_scale (median)':<45} {mae_C:>10.1f} {mape_C:>10.1f}")

    # Chi tiet Method B (tot nhat ve MAPE thong thuong)
    print(f"\n{'=' * 90}")
    print("CHI TIET METHOD B - Per-class effective density")
    print(f"{'=' * 90}")
    print(f"{'FILE':<50} {'CLS':<20} {'GT':>8} {'OLD':>10} {'NEW':>10} {'ERR%':>8}")
    print("-" * 100)
    for c in sorted(calibrated_B, key=lambda x: x["new_rel_err"]):
        short = c["file_name"][:47] + "..." if len(c["file_name"]) > 50 else c["file_name"]
        err_pct = c["new_rel_err"] * 100
        print(f"{short:<50} {c['class_name']:<20} {c['gt_weight_g']:>8.1f} {c['pred_weight_g']:>10.1f} {c['new_pred']:>10.1f} {err_pct:>7.1f}%")

    # Per-class MAE/MAPE Method B
    print(f"\nMethod B per-class:")
    print(f"{'CLASS':<25} {'N':>4} {'MAE':>10} {'MAPE':>10}")
    print("-" * 55)
    for cls in sorted(by_class.keys()):
        cls_entries = [c for c in calibrated_B if c["class_name"] == cls]
        if cls_entries:
            cls_mae = statistics.mean(c["new_abs_err"] for c in cls_entries)
            cls_mape = statistics.mean(c["new_rel_err"] for c in cls_entries) * 100
            print(f"{cls:<25} {len(cls_entries):>4} {cls_mae:>10.1f} {cls_mape:>10.1f}%")

    # ========= GHI FILE CAU HINH MOI =========
    calibration_data = {
        "description": "Calibrated from 38 NutritionVerse-real images (excluded: onion, rice, potato, noodles, pork)",
        "method": "B - Per-class effective density (median of gt_weight / pipeline_volume)",
        "metrics_before": {"MAE_g": round(mae_before, 1), "MAPE_pct": round(mape_before, 1)},
        "metrics_after_B": {"MAE_g": round(mae_B, 1), "MAPE_pct": round(mape_B, 1)},
        "global_scale_factor_A": round(global_scale_median, 6),
        "per_class_effective_density": {k: round(v, 6) for k, v in class_density_median.items()},
        "real_densities_C": REAL_DENSITIES,
        "volume_scale_C": round(vol_scale_median, 6),
    }
    out_json = ROOT / "nv_pipeline" / "calibration_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Calibration data -> {out_json}")


if __name__ == "__main__":
    main()
