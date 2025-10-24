# Quick Reference

Essential commands for VisDrone Toolkit. Bookmark this!

## Setup (First Time)

```bash
# 1. Create & activate virtualenv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install toolkit
pip install -e .

# 4. Test it
python scripts/webcam_demo.py --model fasterrcnn_mobilenet
```

## Training

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

# Fast training (MobileNet + AMP)
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
    --train-img-dir data/train/images \
    --train-ann-dir data/train/annotations \
    --epochs 50
```

## Inference

```bash
# Single image
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input image.jpg

# Directory
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input test_images/

# Video
python scripts/inference.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --input video.mp4
```

## Webcam Demo

```bash
# With trained model
python scripts/webcam_demo.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50

# Without training (COCO weights)
python scripts/webcam_demo.py --model fasterrcnn_mobilenet

# Custom camera & threshold
python scripts/webcam_demo.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --camera 1 \
    --score-threshold 0.7
```

## Evaluation

```bash
# Evaluate model
python scripts/evaluate.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --image-dir data/val/images \
    --annotation-dir data/val/annotations \
    --output-dir eval_results

# Save predictions
python scripts/evaluate.py \
    --checkpoint outputs/my_model/best_model.pth \
    --model fasterrcnn_resnet50 \
    --image-dir data/val/images \
    --annotation-dir data/val/annotations \
    --save-predictions
```

## Convert Annotations

```bash
# To COCO format
python scripts/convert_annotations.py \
    --format coco \
    --image-dir data/images \
    --annotation-dir data/annotations \
    --output annotations_coco.json

# To YOLO format
python scripts/convert_annotations.py \
    --format yolo \
    --image-dir data/images \
    --annotation-dir data/annotations \
    --output-dir data/yolo_labels
```

## Models

| Model                  | Speed      | Accuracy | GPU Memory | Use Case                |
| ---------------------- | ---------- | -------- | ---------- | ----------------------- |
| `fasterrcnn_mobilenet` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐   | 3GB        | Real-time, edge devices |
| `fasterrcnn_resnet50`  | ⭐⭐⭐     | ⭐⭐⭐⭐ | 6GB        | Best balance            |
| `fcos_resnet50`        | ⭐⭐⭐     | ⭐⭐⭐⭐ | 6GB        | Dense objects           |
| `retinanet_resnet50`   | ⭐⭐⭐     | ⭐⭐⭐⭐ | 6GB        | Class imbalance         |

## Common Options

### All Scripts

- `--device cuda` / `--device cpu` - Choose device
- `--help` - Show help message

### Training

- `--amp` - Enable automatic mixed precision (faster!)
- `--batch-size 4` - Batch size (lower if OOM)
- `--lr 0.005` - Learning rate
- `--epochs 50` - Number of epochs
- `--resume checkpoint.pth` - Resume training
- `--save-every 5` - Save checkpoint every N epochs

### Inference

- `--score-threshold 0.5` - Detection confidence threshold
- `--show` - Display results
- `--no-save-viz` - Don't save visualizations

### Webcam

- `--camera 0` - Camera index
- `--score-threshold 0.5` - Detection threshold
- `--width 640 --height 480` - Resolution

## Keyboard Controls

### Webcam Demo

- `q` - Quit
- `s` - Save current frame
- `SPACE` - Pause/Resume

## Python API

```python
from visdrone_toolkit import VisDroneDataset, get_model
from torch.utils.data import DataLoader

# Load dataset
dataset = VisDroneDataset(
    image_dir="data/images",
    annotation_dir="data/annotations"
)

# Create model
model = get_model("fasterrcnn_resnet50", num_classes=12, pretrained=True)

# DataLoader
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
```

## File Structure

```bash
VisDrone-dataset-python-toolkit/
├── visdrone_toolkit/       # Core package
│   ├── dataset.py          # PyTorch Dataset
│   ├── utils.py            # Model factory, metrics
│   ├── visualization.py    # Plotting utilities
│   └── converters/         # Format converters
├── scripts/                # CLI tools
│   ├── train.py
│   ├── inference.py
│   ├── webcam_demo.py
│   ├── evaluate.py
│   └── convert_annotations.py
├── configs/                # Training configs
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── pyproject.toml         # Package config
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch-size 2

# Enable AMP
--amp

# Use smaller model
--model fasterrcnn_mobilenet
```

### Slow Training

```bash
# Enable AMP
--amp

# Increase batch size
--batch-size 8

# Use faster model
--model fasterrcnn_mobilenet
```

### CUDA Not Available

```bash
# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Tips

1. **Always use AMP** on modern GPUs (`--amp`)
2. **Start with MobileNet** for quick experiments
3. **Monitor training curves** in `outputs/training_curves.png`
4. **Save checkpoints frequently** (`--save-every 5`)
5. **Test with webcam** before full training

## Next Steps

- 📖 Read `README.md` for detailed documentation
- 🚀 See `scripts/README.md` for script examples
- ⚙️ Check `configs/README.md` for configuration guide
- 🧪 Run `make test` to verify installation

## Links

- **GitHub:** <https://github.com/dronefreak/VisDrone-dataset-python-toolkit>
- **VisDrone Dataset:** <https://github.com/VisDrone/VisDrone-Dataset>
- **Issues:** <https://github.com/dronefreak/VisDrone-dataset-python-toolkit/issues>
