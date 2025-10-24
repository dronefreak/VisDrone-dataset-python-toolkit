# Configuration Files

YAML configuration files for training VisDrone object detection models.

## Directory Structure

```bash
configs/
├── default_config.yaml           # Default training configuration
└── model_configs/
    ├── fasterrcnn_resnet50.yaml  # Faster R-CNN with ResNet50-FPN
    ├── fasterrcnn_mobilenet.yaml # Faster R-CNN with MobileNetV3-Large-FPN
    ├── fcos_resnet50.yaml        # FCOS (anchor-free)
    └── retinanet_resnet50.yaml   # RetinaNet with Focal Loss
```

## Usage

### Option 1: Use config file directly (future enhancement)

```bash
# This will be supported in future versions
python scripts/train.py --config configs/model_configs/fasterrcnn_resnet50.yaml
```

### Option 2: Use as reference for command-line arguments

Check the config files to see recommended hyperparameters, then use command-line arguments:

```bash
python scripts/train.py \
    --train-img-dir data/VisDrone2019-DET-train/images \
    --train-ann-dir data/VisDrone2019-DET-train/annotations \
    --val-img-dir data/VisDrone2019-DET-val/images \
    --val-ann-dir data/VisDrone2019-DET-val/annotations \
    --model fasterrcnn_resnet50 \
    --epochs 50 \
    --batch-size 4 \
    --lr 0.005 \
    --amp \
    --output-dir outputs/fasterrcnn_resnet50
```

## Model Configurations

### Faster R-CNN ResNet50

**Best for:** Balance of speed and accuracy

- **Accuracy:** ⭐⭐⭐⭐
- **Speed:** ⭐⭐⭐
- **GPU Memory:** ~6GB (batch_size=4)
- **Recommended for:** General use, production deployments

```yaml
model: fasterrcnn_resnet50
batch_size: 4
learning_rate: 0.005
epochs: 50
```

### Faster R-CNN MobileNet

**Best for:** Fast inference, limited GPU memory, edge devices

- **Accuracy:** ⭐⭐⭐
- **Speed:** ⭐⭐⭐⭐⭐
- **GPU Memory:** ~3GB (batch_size=8)
- **Recommended for:** Real-time applications, webcam, drones

```yaml
model: fasterrcnn_mobilenet
batch_size: 8
learning_rate: 0.005
epochs: 50
```

### FCOS ResNet50

**Best for:** Dense object detection, anchor-free approach

- **Accuracy:** ⭐⭐⭐⭐
- **Speed:** ⭐⭐⭐
- **GPU Memory:** ~6GB (batch_size=4)
- **Recommended for:** Scenes with many overlapping objects

```yaml
model: fcos_resnet50
batch_size: 4
learning_rate: 0.01
epochs: 50
```

### RetinaNet ResNet50

**Best for:** Handling class imbalance with Focal Loss

- **Accuracy:** ⭐⭐⭐⭐
- **Speed:** ⭐⭐⭐
- **GPU Memory:** ~6GB (batch_size=4)
- **Recommended for:** Datasets with rare classes

```yaml
model: retinanet_resnet50
batch_size: 4
learning_rate: 0.01
epochs: 50
```

## Hyperparameter Guide

### Batch Size

- **GPU < 8GB:** batch_size = 2-4
- **GPU 8-16GB:** batch_size = 4-8
- **GPU > 16GB:** batch_size = 8-16

Larger batch sizes generally improve training stability but require more memory.

### Learning Rate

- **Faster R-CNN:** 0.005 (with SGD)
- **FCOS/RetinaNet:** 0.01 (higher LR for anchor-free)
- **Rule of thumb:** LR scales with batch size (2x batch = 2x LR)

### Epochs

- **Quick test:** 10-20 epochs
- **Production:** 50-100 epochs
- **Best results:** 100+ epochs with early stopping

### AMP (Automatic Mixed Precision)

- **Recommended:** Always use `--amp` on modern GPUs (RTX series, A100, etc.)
- **Speed boost:** 1.5-2x faster training
- **Memory savings:** ~30-40% less GPU memory

## Creating Custom Configs

Copy an existing config and modify:

```bash
cp configs/model_configs/fasterrcnn_resnet50.yaml configs/my_config.yaml
# Edit my_config.yaml with your settings
```

### Common customizations

**Change dataset paths:**

```yaml
dataset:
  train_img_dir: "path/to/your/train/images"
  train_ann_dir: "path/to/your/train/annotations"
```

**Adjust for limited GPU memory:**

```yaml
training:
  batch_size: 2
  amp: true
```

**Longer training:**

```yaml
training:
  epochs: 100
  lr_scheduler:
    step_size: 30
```

**Fine-tuning pretrained model:**

```yaml
model:
  pretrained: true
  trainable_backbone_layers: 1 # Freeze more layers

training:
  learning_rate: 0.001 # Lower LR for fine-tuning
```

## Tips

1. **Start with default configs** - They're tuned for good results
2. **Use AMP** - Free speed boost on modern GPUs
3. **Monitor training** - Check `outputs/training_curves.png`
4. **Validate regularly** - Use validation set to prevent overfitting
5. **Save checkpoints** - Use `--save-every 5` to save every 5 epochs
6. **Resume training** - Use `--resume checkpoint.pth` if interrupted

## Troubleshooting

**Out of memory?**

- Reduce batch_size
- Enable AMP
- Use fasterrcnn_mobilenet
- Reduce num_workers

**Training too slow?**

- Enable AMP
- Increase batch_size
- Use fewer workers
- Use fasterrcnn_mobilenet

**Poor accuracy?**

- Train longer (more epochs)
- Use larger model (resnet50 vs mobilenet)
- Check learning rate
- Verify dataset quality
