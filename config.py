import os
import json

# PATHS
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============ DATA — RESIZED (OFFLINE) ============
# Bật USE_RESIZED_DATA = True sau khi chạy: python offline_resize.py
USE_RESIZED_DATA = True   # True → dùng data đã resize sẵn đọc từ ổ đĩa (nhanh hơn)

# FoodSemSeg: ảnh + mask PNG (semantic), category_info.json
DATA_DIR_ORIG = os.path.join(PROJECT_ROOT, "data", "FoodSemSeg")
DATA_DIR_RESIZED = os.path.join(PROJECT_ROOT, "data", "FoodSemSeg_512x512")
DATA_DIR = DATA_DIR_RESIZED if USE_RESIZED_DATA else DATA_DIR_ORIG

CATEGORY_INFO_PATH = os.path.join(DATA_DIR, "category_info.json")
CATEGORY_INFO_FALLBACK = os.path.join(PROJECT_ROOT, "config", "category_info_22.json")

# Đọc cấu hình từ category_info.json (fallback config/ nếu data/ chưa có)
_path = CATEGORY_INFO_PATH if os.path.isfile(CATEGORY_INFO_PATH) else CATEGORY_INFO_FALLBACK
with open(_path, "r", encoding="utf-8", errors="replace") as f:
    category_info = json.load(f)

# Cấu trúc: train/images, train/masks | test/images, test/masks
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATA_DIR, "train", "masks")

VAL_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
VAL_LABEL_DIR = os.path.join(DATA_DIR, "test", "masks")

PRETRAINED_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs", "train")

# Chọn pretrained mode: True = dùng ImageNet (khuyến nghị), False = dùng PRETRAINED_PATH
USE_IMAGENET = True

# MODEL
OLD_NUM_CLASSES = 104  # FoodSeg103 pretrained (104 = 103 food + 1 bg)

# Số class từ category_info: 29 = 28 foreground + 1 background
NUM_CLASSES = category_info["num_classes"]

# Tên class: background + danh sách theo thứ tự id (sequential 1..N)
categories = sorted(category_info["categories"], key=lambda x: x["id"])
CLASS_NAMES = ["background"] + [c["name"] for c in categories]

# Mask dùng sequential ID (0..N) — không cần mapping
RAW_TO_SEQ = None

# Class weights: tính từ tần suất (median). Class IoU thấp tăng weight qua CLASS_WEIGHT_BOOST.
# Factor clamp [1.0, 2.0]. Class hiếm (tofu 11) boost 2.0.
CLASS_WEIGHT_BOOST = {
    11: 2.0,  # tofu
}

# ============ TRAINING — Train từ đầu ============
SEED = 42
IMG_SIZE = (512, 512)  # (W, H)
BATCH_SIZE = 4
VAL_BATCH_SIZE = 8
NUM_WORKERS = 4
USE_AMP = False
GRAD_CLIP_MAX_NORM = 0.5
LABEL_SMOOTHING = 0.0

# Phase 1 — Linear probing (freeze encoder)
P1_LR = 1e-3
P1_EPOCHS = 100
P1_PATIENCE = 8

# Phase 2 — Fine-tune full
P2_LR_ENCODER = 1e-5
P2_LR_DECODER = 5e-5
P2_LR_HEAD = 2e-4
P2_WEIGHT_DECAY = 3e-4
P2_EPOCHS = 100
P2_PATIENCE = 15
P2_USE_EMA = True
P2_EMA_DECAY = 0.999
P2_USE_COSINE = False
P2_WARMUP_EPOCHS = 2
P2_SCHEDULER_PATIENCE = 4
P2_SCHEDULER_FACTOR = 0.5

# TTA (final eval)
TTA_SCALES = [1.0]
TTA_HFLIP = True
TTA_VFLIP = True