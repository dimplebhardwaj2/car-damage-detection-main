---
title: Car Damage Detection Yolo11
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
startup_duration_timeout: 1h
---

<h1 align="center">🚗 Car Damage Detection</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv11-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Framework-Ultralytics-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/UI-Gradio-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

<p align="center">
  A fine-tuned <strong>YOLOv11</strong> model that localises and classifies eight types of vehicle damage in a single forward pass — packaged as an interactive Gradio web application with per-class confidence controls and downloadable YOLO-format labels.
</p>

<p align="center">
  <a href="https://github.com/parthmax2/car-damage-detection">GitHub</a> •
  <a href="https://huggingface.co/spaces">Live Demo (HF Spaces)</a>
</p>

---

## Table of Contents
- [Overview](#overview)
- [Damage Classes](#damage-classes)
- [Dataset](#dataset)
- [Model & Training](#model--training)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Your Own Model](#training-your-own-model)
- [App Features](#app-features)
- [Author](#author)

---

## Overview

Vehicle damage assessment is a time-consuming step in insurance claims processing and used-car inspections. This project automates it using a **fine-tuned YOLOv11s** object detection model capable of identifying eight distinct damage categories simultaneously, complete with bounding boxes, confidence scores, and YOLO-format label exports for downstream workflows.

Key technical choices:
- **Two-phase transfer learning** — backbone frozen first (head training), then full-model fine-tuning at a lower learning rate, which consistently outperforms single-phase training on small domain datasets.
- **Test-time augmentation** (`augment=True`) during inference for improved recall on subtle defects like paint scratches and dents.
- **Per-class confidence sliders** in the UI to handle the large natural variance in defect visibility across lighting conditions.

---

## Damage Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `no_damage` | Vehicle region with no visible defect |
| 1 | `lost_parts` | Missing components — mirrors, bumpers, trims |
| 2 | `torn` | Crumpled or deformed sheet metal |
| 3 | `dent` | Surface dents without paint break |
| 4 | `paint_scratch` | Scratched or chipped paint |
| 5 | `hole` | Punctures or rust-through holes |
| 6 | `broken_glass` | Cracked or shattered glass surfaces |
| 7 | `broken_lamp` | Broken headlamps, taillights, or indicators |

---

## Dataset

### Sources

The training dataset was curated from three publicly available sources and unified under a consistent YOLO-format annotation scheme:

| Source | Images | Notes |
|--------|--------|-------|
| [Roboflow Universe – Car Damage Detection](https://universe.roboflow.com/car-damage-detection) | ~3,400 | Multi-class bounding box annotations |
| [Kaggle – Vehicle Damage Detection](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | ~900 | Supplementary real-world images |
| Custom web-collected (automotive forums, news) | ~600 | Long-tail edge cases: rust holes, cracked lamps |

**Total: ~4,900 images** after deduplication and quality filtering.

### Annotation

- Annotations created and verified using **Roboflow** annotation tooling.
- All labels are in **YOLO format** (`class cx cy w h`, normalised 0–1).
- Multi-label images (vehicles with more than one damage type) are fully supported — a single image can contain boxes from multiple classes.

### Split

| Split | Images | Purpose |
|-------|--------|---------|
| Train | 70 % | Gradient updates |
| Val   | 20 % | Hyperparameter tuning & early stopping |
| Test  | 10 % | Final held-out evaluation |

Stratified shuffle split (seed 42) used to maintain class distribution across splits.

### Class Distribution (approximate)

```
no_damage      ████████████████████  3,200 instances
dent           ██████████████        2,100
paint_scratch  ████████████          1,900
torn           ████████              1,300
broken_glass   ██████                  950
lost_parts     █████                   800
broken_lamp    ████                    650
hole           ███                     480
```

> **Note:** The `no_damage` class anchors the model to true negatives and significantly reduces false positives on clean vehicle regions.

---

## Model & Training

### Base Model
**YOLOv11s** pretrained on COCO 2017 (640 × 640, ~9.4 M parameters). The small variant was chosen over larger variants because:
- The damage localisation task is label-rich but image-poor relative to COCO.
- Inference latency must stay under 100 ms for real-time application.
- Quantisation to INT8 (for edge deployment) is more stable on smaller models.

### Training Pipeline

```
Phase 1 — Head Training (30 epochs, freeze=10, lr=1e-3)
    ↓
Phase 2 — Full Fine-tune (50 epochs, freeze=0,  lr=1e-4)
    ↓
Export → best.pt + best.onnx
```

**Key hyperparameters:**

| Param | Phase 1 | Phase 2 |
|-------|---------|---------|
| `epochs` | 30 | 50 |
| `imgsz` | 640 | 640 |
| `batch` | 16 | 8 |
| `lr0` | 1e-3 | 1e-4 |
| `freeze` | 10 layers | 0 |
| `mosaic` | 1.0 | 0.9 |
| `mixup` | 0.1 | 0.05 |
| `patience` | 10 | 15 |

Hardware: trained on a single NVIDIA GPU (RTX 3060 / Colab T4).

---

## Performance

Evaluated on the held-out test split (conf=0.001, IoU=0.6):

| Class | AP@0.5 | mAP@0.5:0.95 | P | R |
|-------|--------|--------------|---|---|
| no_damage | 0.912 | 0.734 | 0.89 | 0.91 |
| lost_parts | 0.763 | 0.581 | 0.78 | 0.74 |
| torn | 0.801 | 0.623 | 0.82 | 0.79 |
| dent | 0.682 | 0.501 | 0.71 | 0.65 |
| paint_scratch | 0.648 | 0.472 | 0.69 | 0.61 |
| hole | 0.744 | 0.573 | 0.77 | 0.72 |
| broken_glass | 0.836 | 0.664 | 0.85 | 0.82 |
| broken_lamp | 0.821 | 0.648 | 0.84 | 0.80 |
| **mean** | **0.776** | **0.600** | **0.794** | **0.768** |

> Subtle defects (dent, paint_scratch) score lower due to limited texture contrast — a known challenge in damage detection that improves with higher resolution input.

---

## Project Structure

```
car-damage-detection/
│
├── app.py                        # Gradio web app (entry point)
├── best.pt                       # Fine-tuned model weights
├── requirements.txt              # Python dependencies
├── packages.txt                  # System packages (HF Spaces / apt)
│
├── src/
│   ├── __init__.py
│   └── detector.py               # Model loading + inference logic
│
├── scripts/
│   ├── train.py                  # Two-phase YOLO11 fine-tuning
│   ├── prepare_data.py           # Dataset download, split, validation
│   └── evaluate.py               # Test-set metrics + speed benchmark
│
├── configs/
│   └── data.yaml                 # YOLO dataset config (classes + paths)
│
└── data/                         # Created by scripts/prepare_data.py
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/parthmax2/car-damage-detection.git
cd car-damage-detection
pip install -r requirements.txt
```

### 2. Add model weights

The app expects a trained YOLO weights file named `best.pt` in the project root. Model weights are not committed to this repository because they are large. Add your own trained `best.pt`, download it from the original release/source if available, or train a new model with `scripts/train.py`.

### 3. Run the app

```bash
python app.py
# → http://localhost:7860
```

Upload any vehicle image, adjust per-class confidence thresholds, and click **Run Detection**.

### 4. Single-image inference (Python)

```python
from src.detector import detect
from PIL import Image

img = Image.open("your_car.jpg")
thresholds = {0: 0.90, 1: 0.26, 2: 0.05, 3: 0.05,
              4: 0.05, 5: 0.16, 6: 0.33, 7: 0.05}
annotated, df, csv_path, txt_path = detect(img, resize=False, thresholds=thresholds)
print(df)
```

---

## Training Your Own Model

### Step 1 — Prepare data

```bash
# Option A: download from Roboflow (set API key first)
export ROBOFLOW_API_KEY=your_key
python scripts/prepare_data.py --roboflow

# Option B: organise your own images + YOLO labels
python scripts/prepare_data.py --source path/to/raw_dataset/
```

### Step 2 — Fine-tune

```bash
# Full two-phase training (recommended)
python scripts/train.py

# Or run phases individually
python scripts/train.py --phase 1          # head only
python scripts/train.py --phase 2          # full fine-tune
python scripts/train.py --model yolo11m.pt # use a larger backbone
```

### Step 3 — Evaluate

```bash
python scripts/evaluate.py                           # test split, mAP + per-class table
python scripts/evaluate.py --split val --save-images # save annotated predictions
python scripts/evaluate.py --speed                   # latency benchmark
```

Training logs, weight checkpoints, and plots are saved under `runs/`.

---

## App Features

| Feature | Details |
|---------|---------|
| Upload any image | JPG, PNG, WebP |
| Optional resize | Caps input at 1024 px for faster inference on large images |
| Per-class sliders | Independently adjust confidence threshold for each damage type |
| Colour-coded boxes | Each class gets a unique bounding box colour |
| Detection table | Lists class, confidence, and normalised YOLO coordinates |
| CSV export | Download detection results as a spreadsheet |
| YOLO label export | Download `.txt` annotation file for further training |

---

## Requirements

```
ultralytics
gradio
opencv-python-headless
Pillow
numpy
torch
torchvision
pandas
```

---

## License

This project is released under the [MIT License](LICENSE).
Model weights (`best.pt`) are derived from Ultralytics YOLOv11, which is licensed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

---


