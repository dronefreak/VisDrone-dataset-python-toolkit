# VisDrone Toolkit 2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Modern PyTorch-based toolkit for the VisDrone dataset with state-of-the-art object detection models.

## 🚀 What's New in 2.0

- ✅ **PyTorch-first** - Native PyTorch Dataset, modern torchvision models
- ✅ **Multiple architectures** - Faster R-CNN, FCOS, RetinaNet (ResNet50 & MobileNet)
- ✅ **Real-time webcam demo** - Test detection instantly with your webcam
- ✅ **Modern formats** - COCO & YOLO converters (not just VOC)
- ✅ **Production ready** - CLI tools, proper packaging, comprehensive tests
- ✅ **Easy installation** - Virtualenv-based, one-command setup

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models](#-models)
- [Documentation](#-documentation)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

## ✨ Features

### Core Components

- **PyTorch Dataset** - Native VisDrone format support with automatic filtering
- **Model Zoo** - 4 pre-configured detection models (Faster R-CNN, FCOS, RetinaNet)
- **Format Converters** - Convert to COCO or YOLO with validation
- **Visualization Tools** - Publication-ready plots and detection overlays
- **CLI Tools** - Train, evaluate, and infer with simple commands

### Training

- Automatic mixed precision (AMP) for faster training
- Multi-GPU support
- Learning rate scheduling
- Checkpointing and resuming
- Training curve visualization

### Inference

- Real-time webcam detection
- Batch inference on images/videos
- Configurable confidence thresholds
- FPS measurement and benchmarking

## 🎯 Quick Start

```bash
# 1. Install
git clone https://github.com/dronefreak/VisDrone-dataset-python-toolkit.git
cd VisDrone-dataset-python-toolkit
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# 2. Test with webcam (no training needed!)
python scripts/webcam_demo.py --model fasterrcnn_mobilenet

# 3. Train on VisDrone dataset
python scripts/train.py \
    --train-img-dir data/VisDrone2019-DET-train/images \
    --train-ann-dir data/VisDrone2019-DET-train/annotations \
    --val-img-dir data/VisDrone2019-DET-val/images \
    --val-ann-dir data/VisDrone2019-DET-val/annotations \
    --model fasterrcnn_resnet50 \
    --epochs 50 \
    --batch-size 4 \
    --amp \
    --output-dir outputs/my_model

# 4. Run inference
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input test_images/ \
    --output-dir results
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### Step 2: Install PyTorch

**For GPU (CUDA 11.8):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install VisDrone Toolkit

```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With COCO evaluation
pip install -e ".[coco]"
```

### Step 4: Download VisDrone Dataset

Download from [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) and extract:

```bash
data/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## 🎓 Usage

### Training

```bash
# Basic training
python scripts/train.py \
    --train-img-dir data/train/images \
    --train-ann-dir data/train/annotations \
    --val-img-dir data/val/images \
    --val-ann-dir data/val/annotations \
    --model fasterrcnn_resnet50 \
    --epochs 50 \
    --batch-size 4 \
    --output-dir outputs/my_model

# Fast training with AMP
python scripts/train.py \
    --train-img-dir data/train/images \
    --train-ann-dir data/train/annotations \
    --model fasterrcnn_mobilenet \
    --epochs 30 \
    --batch-size 8 \
    --amp \
    --output-dir outputs/mobilenet

# Resume training
python scripts/train.py \
    --resume outputs/my_model/checkpoint_epoch_20.pth \
    --epochs 50
```

### Inference

```bash
# Single image
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input image.jpg

# Directory of images
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input test_images/ \
    --output-dir results

# Video file
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input video.mp4 \
    --output-dir results
```

### Webcam Demo

```bash
# With trained model
python scripts/webcam_demo.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50

# With pretrained weights (no training needed!)
python scripts/webcam_demo.py --model fasterrcnn_mobilenet

# Custom camera and threshold
python scripts/webcam_demo.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --camera 1 \
    --score-threshold 0.7
```

**Controls:**

- `q` - Quit
- `s` - Save current frame
- `SPACE` - Pause/Resume

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --image-dir data/val/images \
    --annotation-dir data/val/annotations \
    --output-dir eval_results
```

### Format Conversion

```bash
# Convert to COCO format
python scripts/convert_annotations.py \
    --format coco \
    --image-dir data/images \
    --annotation-dir data/annotations \
    --output annotations_coco.json

# Convert to YOLO format
python scripts/convert_annotations.py \
    --format yolo \
    --image-dir data/images \
    --annotation-dir data/annotations \
    --output-dir data/yolo_labels
```

### Python API

```python
from visdrone_toolkit import VisDroneDataset, get_model
from torch.utils.data import DataLoader

# Load dataset
dataset = VisDroneDataset(
    image_dir="data/images",
    annotation_dir="data/annotations",
    filter_ignored=True,
    filter_crowd=True,
)

# Create model
model = get_model("fasterrcnn_resnet50", num_classes=12, pretrained=True)

# DataLoader
from visdrone_toolkit.utils import collate_fn
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

# Training loop
model.train()
for images, targets in loader:
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    # ... backpropagation
```

## 🤖 Models

| Model                      | Speed      | Accuracy | GPU Memory | Best For                    |
| -------------------------- | ---------- | -------- | ---------- | --------------------------- |
| **Faster R-CNN ResNet50**  | ⭐⭐⭐     | ⭐⭐⭐⭐ | 6GB        | General use, best balance   |
| **Faster R-CNN MobileNet** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐   | 3GB        | Real-time, edge devices     |
| **FCOS ResNet50**          | ⭐⭐⭐     | ⭐⭐⭐⭐ | 6GB        | Dense objects, anchor-free  |
| **RetinaNet ResNet50**     | ⭐⭐⭐     | ⭐⭐⭐⭐ | 6GB        | Class imbalance, Focal Loss |

### Performance Benchmarks

On VisDrone2019-DET-val (RTX 3090, batch_size=4):

| Model                  | mAP@50 | FPS | Training Time (50 epochs) |
| ---------------------- | ------ | --- | ------------------------- |
| Faster R-CNN ResNet50  | ~35%   | 18  | ~8 hours                  |
| Faster R-CNN MobileNet | ~30%   | 45  | ~6 hours                  |
| FCOS ResNet50          | ~33%   | 16  | ~8 hours                  |
| RetinaNet ResNet50     | ~34%   | 17  | ~8 hours                  |

**_Note: Results depend on training configuration and dataset split_**

## 📚 Documentation

- **[Installation Guide](INSTALL.md)** - Detailed setup instructions
- **[Quick Reference](QUICKSTART.md)** - Command cheatsheet
- **[Scripts Documentation](scripts/README.md)** - CLI tools usage
- **[Configuration Guide](configs/README.md)** - Training configs
- **[Test Documentation](tests/README.md)** - Running tests
- **[Contributing Guide](CONTRIBUTING.md)** - Development workflow
- **[Changelog](CHANGELOG.md)** - Version history

## 💡 Examples

### Example 1: Quick Webcam Test

```bash
# No training needed - uses COCO pretrained weights
python scripts/webcam_demo.py --model fasterrcnn_mobilenet
```

### Example 2: Train Custom Model

```bash
python scripts/train.py \
    --train-img-dir data/VisDrone2019-DET-train/images \
    --train-ann-dir data/VisDrone2019-DET-train/annotations \
    --val-img-dir data/VisDrone2019-DET-val/images \
    --val-ann-dir data/VisDrone2019-DET-val/annotations \
    --model fasterrcnn_resnet50 \
    --epochs 50 \
    --batch-size 4 \
    --amp \
    --output-dir outputs/custom_model
```

### Example 3: Batch Inference

```bash
python scripts/inference.py \
    --checkpoint outputs/custom_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input drone_videos/*.mp4 \
    --output-dir results/batch_inference
```

### Example 4: Convert to YOLO for YOLOv8

```bash
python scripts/convert_annotations.py \
    --format yolo \
    --image-dir data/VisDrone2019-DET-train/images \
    --annotation-dir data/VisDrone2019-DET-train/annotations \
    --output-dir data/yolo/train/labels

# Then use with YOLOv8
# yolo train data=data/yolo/dataset.yaml model=yolov8n.pt
```

## 🔧 Advanced Usage

### Custom Data Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

dataset = VisDroneDataset(
    image_dir="data/images",
    annotation_dir="data/annotations",
    transforms=transform,
)
```

### Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model, device_ids=[local_rank])
```

### Export to ONNX

```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['boxes', 'labels', 'scores']
)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Code style guidelines
- Pull request process
- Issue reporting

### Quick Contribution Guide

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/VisDrone-dataset-python-toolkit.git
cd VisDrone-dataset-python-toolkit

# 2. Setup development environment
make setup-venv
source venv/bin/activate
make install-dev

# 3. Create feature branch
git checkout -b feature/amazing-feature

# 4. Make changes and test
make format
make lint
make test

# 5. Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# 6. Open Pull Request
```

## 📖 Citation

If you use this toolkit in your research, please cite:

```bibtex
@misc{visdrone_toolkit_2025,
  author = {Saksena, Saumya Kumaar},
  title = {VisDrone Toolkit 2.0: Modern PyTorch Implementation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dronefreak/VisDrone-dataset-python-toolkit}
}
```

And the original VisDrone dataset:

```bibtex
@article{zhu2018visdrone,
  title={Vision Meets Drones: A Challenge},
  author={Zhu, Pengfei and Wen, Longyin and Bian, Xiao and Ling, Haibin and Hu, Qinghua},
  journal={arXiv preprint arXiv:1804.07437},
  year={2018}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VisDrone Team** - For creating and maintaining the dataset
- **PyTorch & Torchvision** - For the excellent deep learning framework
- **Contributors** - Everyone who has contributed to this project (just me for the moment)

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/dronefreak/VisDrone-dataset-python-toolkit/issues)
- **Discussions:** [GitHub Discussions](https://github.com/dronefreak/VisDrone-dataset-python-toolkit/discussions)
- **Email:** <your.email@example.com>

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

## 🗺️ Roadmap

- [ ] Support for VisDrone video tasks
- [ ] Integration with Weights & Biases
- [ ] TensorRT optimization
- [ ] Docker images
- [ ] More model architectures (DETR, YOLOv8)
- [ ] Mobile deployment guide

## 📊 Project Stats

- **Version:** 2.0.0
- **Python:** 3.8+
- **PyTorch:** 2.0+
- **Tests:** 66 unit tests
- **Coverage:** >80%
- **Models:** 4 architectures
- **Scripts:** 5 CLI tools

---
