# VisDrone Toolkit

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

PyTorch toolkit for the [VisDrone aerial detection dataset](https://github.com/VisDrone/VisDrone-Dataset). Supports 33 models (4 torchvision + 29 YOLO), end-to-end training, evaluation, and inference.

<p align="center">
  <img src="../assets/visdrone_showcase.gif" alt="VisDrone Detection Demo" width="150%">
</p>

_Example: YOLO26x predictions on VisDrone video sequences using Soft-NMS (confidence threshold = 0.5)._

---

## Installation

```bash
git clone https://github.com/dronefreak/VisDrone-dataset-python-toolkit.git
cd VisDrone-dataset-python-toolkit
python -m venv venv && source venv/bin/activate
pip install -e .                 # basic
pip install -e ".[dev]"          # with dev tools
```

**Dataset layout** (download from [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)):

```bash
data/
├── VisDrone2019-DET-train/images/  annotations/
└── VisDrone2019-DET-val/images/    annotations/
```

---

## Models

| Model                                          | Type        | Notes                           |
| ---------------------------------------------- | ----------- | ------------------------------- |
| `fasterrcnn_resnet50` / `fasterrcnn_mobilenet` | Torchvision | Best accuracy / lightweight     |
| `fcos_resnet50`                                | Torchvision | Anchor-free                     |
| `retinanet_resnet50`                           | Torchvision | Focal loss, class imbalance     |
| `yolov8n/s/m/l/x`                              | YOLO v8     | Recommended for new experiments |
| `yolov9c/e/m`                                  | YOLO v9     | Programmable gradient info      |
| `yolov10n/s/m/b/l/x`                           | YOLO v10    | NMS-free inference              |
| `yolo11n/s/m/l/x`                              | YOLO 11     | 2024 C3k2+C2PSA architecture    |
| `yolo26n/s/m/l/x`                              | YOLO 26     | 2025, best efficiency           |

```bash
python scripts/train.py --available-models   # list all 33 models
```

---

## Usage

### Train

```bash
# Torchvision (Faster R-CNN)
python scripts/train.py \
    --train-img-dir data/VisDrone2019-DET-train/images \
    --train-ann-dir data/VisDrone2019-DET-train/annotations \
    --val-img-dir   data/VisDrone2019-DET-val/images \
    --val-ann-dir   data/VisDrone2019-DET-val/annotations \
    --model fasterrcnn_resnet50 --epochs 200 --batch-size 2 \
    --amp --augmentation --multiscale --small-anchors \
    --lr 0.005 --lr-schedule multistep --lr-milestones 60 80 \
    --output-dir outputs/fasterrcnn_200ep

# YOLO (delegates to Ultralytics engine)
python scripts/train.py \
    --train-img-dir data/VisDrone2019-DET-train/images \
    --train-ann-dir data/VisDrone2019-DET-train/annotations \
    --val-img-dir   data/VisDrone2019-DET-val/images \
    --val-ann-dir   data/VisDrone2019-DET-val/annotations \
    --model yolov8n --epochs 200 --batch-size 16 --amp \
    --output-dir outputs/yolov8n_200ep
```

Weights are saved as `best.pt` and `last.pt` inside `--output-dir`.

> **YOLO note:** `--multiscale`, `--small-anchors`, `--lr-schedule`, and `--accumulation-steps` are ignored for YOLO models — these are handled internally by Ultralytics. `--num-classes` is automatically clamped to 11 (VisDrone's 11 real classes).

### Evaluate

```bash
# Torchvision — P/R/F1 + optional pycocotools mAP
python scripts/evaluate.py \
    --checkpoint outputs/fasterrcnn_200ep/best.pt \
    --model fasterrcnn_resnet50 \
    --image-dir data/VisDrone2019-DET-val/images \
    --annotation-dir data/VisDrone2019-DET-val/annotations

# YOLO — mAP@0.5 and mAP@0.5:0.95 via Ultralytics val engine
python scripts/evaluate.py \
    --checkpoint outputs/yolov8n_200ep/yolov8n/weights/best.pt \
    --model yolov8n \
    --image-dir data/VisDrone2019-DET-val/images \
    --annotation-dir data/VisDrone2019-DET-val/annotations
```

Outputs a rich per-class metrics table and saves `eval_outputs/metrics.json`.

### Inference

```bash
# Images / directory / video — auto-detected from file extension
python scripts/inference.py \
    --checkpoint outputs/yolov8n_200ep/yolov8n/weights/best.pt \
    --model yolov8n --input data/images/ --output-dir results

python scripts/inference.py \
    --checkpoint outputs/fasterrcnn_200ep/best.pt \
    --model fasterrcnn_resnet50 --input drone_video.mp4 \
    --soft-nms --score-threshold 0.5 --output-dir results
```

### Webcam / Video Demo

```bash
# Webcam (default source=0)
python scripts/webcam_demo.py \
    --checkpoint outputs/yolov8n_200ep/yolov8n/weights/best.pt \
    --model yolov8n

# Video file or RTSP stream
python scripts/webcam_demo.py \
    --checkpoint outputs/fasterrcnn_200ep/best.pt \
    --model fasterrcnn_resnet50 --source drone_video.mp4

# COCO pretrained weights — no VisDrone training needed
python scripts/webcam_demo.py --model fasterrcnn_mobilenet
```

Controls: `q` quit | `s` save frame | `Space` pause

### Format Conversion

```bash
# VisDrone → COCO
python scripts/convert_annotations.py --format coco \
    --image-dir data/images --annotation-dir data/annotations \
    --output annotations_coco.json

# VisDrone → YOLO
python scripts/convert_annotations.py --format yolo \
    --image-dir data/images --annotation-dir data/annotations \
    --output-dir data/yolo_labels
```

### Python API

```python
from visdrone_toolkit import VisDroneDataset, get_model
from visdrone_toolkit.utils import collate_fn
from torch.utils.data import DataLoader

dataset = VisDroneDataset(
    image_dir="data/images",
    annotation_dir="data/annotations",
    filter_ignored=True,
    filter_crowd=True,
)
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
model = get_model("fasterrcnn_resnet50", num_classes=12, pretrained=True)
```

---

## Performance

Faster R-CNN ResNet50, VisDrone2019-DET-val (200 epochs, RTX 4070 Super):

| Metric    | Score               |
| --------- | ------------------- |
| F1        | 66.7%               |
| Precision | 71.0%               |
| Recall    | 62.9%               |
| Speed     | 18 FPS (55ms/image) |

YOLO v8n after 1 epoch (untrained baseline): mAP@0.5 = 0.119, mAP@0.5:0.95 = 0.062.

---

## Development

```bash
make format lint test       # format + lint + run tests
python -m pytest            # 203 tests, ~63% coverage
```

Pre-commit hooks: Black, Ruff, isort, mypy.

---

## Citation

```bibtex
@misc{visdrone_toolkit_2025,
  author = {Saksena, Saumya Kumaar},
  title  = {VisDrone Toolkit 2.0},
  year   = {2025},
  url    = {https://github.com/dronefreak/VisDrone-dataset-python-toolkit}
}

@article{zhu2018visdrone,
  title   = {Vision Meets Drones: A Challenge},
  author  = {Zhu, Pengfei and Wen, Longyin and Bian, Xiao and Ling, Haibin and Hu, Qinghua},
  journal = {arXiv preprint arXiv:1804.07437},
  year    = {2018}
}
```

---

[Changelog](CHANGELOG.md) · [Issues](https://github.com/dronefreak/VisDrone-dataset-python-toolkit/issues) · Apache 2.0
