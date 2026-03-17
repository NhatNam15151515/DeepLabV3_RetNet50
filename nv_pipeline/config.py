"""
Cấu hình tối thiểu cho nv_pipeline (depth/volume + weight).

Tách riêng khỏi config train segmentation để pipeline có thể chạy độc lập.
"""

from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Thư mục output mặc định cho các kết quả pipeline (depth map, báo cáo, v.v.)
OUTPUT_DIR = PROJECT_ROOT / "nv_pipeline_outputs"


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Cấu hình Tầng 2 (MiDaS depth + volume)
TIER2_CONFIG = {
    "model_type": "DPT_Large",          # hoặc "DPT_Hybrid" nếu muốn nhẹ hơn
    "focal_length_px": 1000.0,
    "reference_distance_cm": 40.0,
    "depth_scale_factor": 1.0,
}


# Cấu hình Tầng 3 (density + weight)
# global_scale_factor calibrated từ 38 ảnh NV-real (Method C: vol_scale median)
TIER3_CONFIG = {
    "density_file": None,
    "default_density": 0.9,             # g/cm³ - fallback cho class chưa calibrate
    "global_scale_factor": 0.042778,    # volume correction factor (MiDaS overestimates ~23x)
}


# Bảng mật độ thực (g/cm³) cho NutritionVerse classes.
# Dùng kết hợp với global_scale_factor để ước lượng weight.
NV_REAL_DENSITIES = {
    "chicken-wing": 1.05,
    "chicken-leg": 1.05,
    "chicken-breast": 1.04,
    "near-whole-chicken": 0.95,
    "lasagna": 1.10,
    "steak-piece": 1.08,
    "costco-egg": 1.03,
    "steak": 1.08,
    "salad-chicken-strip": 1.02,
    "half-bread-loaf": 0.35,
    "plain-toast": 0.35,
    "stack-of-tofu-4pc": 0.95,
    "cucumber-piece": 0.96,
    "carrot": 1.04,
}

