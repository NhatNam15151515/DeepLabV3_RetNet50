# DeepLabV3+ RetNet50 – Food Semantic Segmentation

Phân vùng ngữ nghĩa thực phẩm với DeepLabV3+ (ResNet50) trên dataset FoodSemSeg, kèm pipeline 3 tầng (seg → depth → weight) để ước lượng trọng lượng.

## Setup

```bash
pip install -r requirements.txt
```

**Dữ liệu:** Tạo thư mục `data/FoodSemSeg_512x512/` với cấu trúc:
- `category_info.json` — copy từ `config/category_info_22.json`
- `train/images/`, `train/masks/`
- `test/images/`, `test/masks/`

## Cấu trúc dự án

```
├── config.py, dataset.py, model_setup.py, metrics.py, utils.py, postprocess.py
├── config/                    # category_info mẫu (22 class, 29 class)
├── nv_pipeline/              # Pipeline 3 tầng seg → depth → weight
│   ├── tier1_segmentation.py
│   ├── tier2_depth_volume.py
│   ├── tier3_weight_estimation.py
│   └── pipeline.py
└── scripts/
    ├── training/              # train.py, finetune_from_checkpoint.py, eval_ensemble.py
    ├── api/                   # api.py (22 class), api_exp19_29class.py (29 class), test_api.py
    ├── data_tools/            # compute_class_stats, visualize_all_data, offline_resize
    ├── nv_pipeline/          # calibrate_tier23, test_nv_subset_pipeline
    └── generate_report.py     # Sinh báo cáo .docx
```

## Training

**Train từ đầu (2 phase):**
```bash
python scripts/training/train.py
```
- Phase 1: freeze encoder, train decoder + head (LR 1e-3)
- Phase 2: unfreeze toàn bộ, differential LR (encoder 1e-5, head 2e-4)
- Checkpoint: `runs/train/expXX/weights/best.pth`

**Fine-tune từ checkpoint:**
```bash
python scripts/training/finetune_from_checkpoint.py --checkpoint runs/train/exp21/weights/best.pth
```

**Đánh giá ensemble:**
```bash
python scripts/training/eval_ensemble.py --checkpoints runs/train/exp21/weights/best.pth --tta
```

## API

**Khởi động (22 class):**
```bash
uvicorn scripts.api.api:app --host 0.0.0.0 --port 8000
```

**Legacy 29 class:**
```bash
uvicorn scripts.api.api_exp19_29class:app --host 0.0.0.0 --port 8001
```

**Test:**
```bash
python scripts/api/test_api.py
```

**Đổi checkpoint:** Sửa `API_BEST_PATH` trong `scripts/api/api.py` (mặc định `exp21`).

## Pipeline 3 tầng (nv_pipeline)

| Tầng | Input | Output |
|------|-------|--------|
| Tier 1 | Ảnh RGB | Semantic mask + instance extraction |
| Tier 2 | Ảnh + mask | MiDaS depth → volume (cm³) |
| Tier 3 | Volume + class | Trọng lượng (g) = density × volume × scale |

API đã tích hợp pipeline này; response JSON gồm `detections` (class, instance_count, estimated_weight_g).

## Data tools

- `compute_class_stats.py` — thống kê pixel / ảnh per class
- `visualize_all_data.py` — xem ảnh + mask + legend
- `offline_resize.py` — resize/pad offline tạo `FoodSemSeg_512x512`

## Báo cáo

```bash
python scripts/generate_report.py
```
Sinh file `reports/BaoCao_DeepLabV3_FoodSeg.docx`.

## Cấu hình chính (config.py)

- `USE_IMAGENET = True` — pretrained ImageNet (khuyến nghị)
- `USE_RESIZED_DATA = True` — dùng data đã resize sẵn
- `NUM_CLASSES` — đọc từ `category_info.json`
- `P1_*`, `P2_*` — hyperparams Phase 1, Phase 2
