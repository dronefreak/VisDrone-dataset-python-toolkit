# Installation Guide

Complete guide for setting up the VisDrone Toolkit with virtualenv.

## Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/dronefreak/VisDrone-dataset-python-toolkit.git
cd VisDrone-dataset-python-toolkit

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install package
pip install -e .

# 4. Test with webcam (no training needed!)
python scripts/webcam_demo.py --model fasterrcnn_mobilenet
```

## Detailed Installation

### Prerequisites

- **Python:** 3.8 or higher
- **GPU (recommended):** NVIDIA GPU with CUDA support
- **CPU only:** Works but much slower

Check your Python version:

```bash
python3 --version  # Should be 3.8+
```

### Step 1: Clone Repository

```bash
git clone https://github.com/dronefreak/VisDrone-dataset-python-toolkit.git
cd VisDrone-dataset-python-toolkit
```

### Step 2: Create Virtual Environment

**Why virtualenv?** Keeps dependencies isolated from your system Python.

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Your prompt should now show (venv)
```

### Step 3: Install PyTorch

**For GPU (CUDA):**

```bash
# CUDA 11.8 (most common)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Install VisDrone Toolkit

#### **Option A: Editable install (recommended for development)**

```bash
pip install -e .
```

#### **Option B: Regular install**

```bash
pip install .
```

#### **Option C: With development tools**

```bash
pip install -e ".[dev]"
```

### Step 5: Verify Installation

```bash
# Check if package is installed
python -c "import visdrone_toolkit; print(visdrone_toolkit.__version__)"

# Run a quick test
python scripts/webcam_demo.py --help
```

## Installation Options

### Minimal Installation (Core only)

```bash
pip install -r requirements.txt
pip install -e .
```

### Development Installation (All tools)

```bash
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

### With COCO Evaluation

```bash
pip install -e ".[coco]"
```

### Everything

```bash
pip install -e ".[all]"
```

## Using Makefile (Recommended)

We provide a Makefile for common tasks:

```bash
# Create venv
make setup-venv

# Activate it
source venv/bin/activate

# Install with dev dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Clean build artifacts
make clean
```

See all commands:

```bash
make help
```

## GPU Setup

### Check CUDA Version

```bash
nvidia-smi
```

Look for "CUDA Version: X.X" in the output.

### Install Matching PyTorch

Match PyTorch CUDA version to your system:

| System CUDA | PyTorch Command                                                                    |
| ----------- | ---------------------------------------------------------------------------------- |
| 11.8        | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| 12.1        | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| CPU only    | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`   |

### Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should output:

```python
True
NVIDIA GeForce RTX 3090  # (or your GPU name)
```

## Download VisDrone Dataset

```bash
# Create data directory
mkdir -p data

# Download from official VisDrone website
# https://github.com/VisDrone/VisDrone-Dataset

# Extract files
# Expected structure:
# data/
# ├── VisDrone2019-DET-train/
# │   ├── images/
# │   └── annotations/
# └── VisDrone2019-DET-val/
#     ├── images/
#     └── annotations/
```

## First Run

### Test with Webcam (No Training!)

```bash
python scripts/webcam_demo.py --model fasterrcnn_mobilenet
```

This uses pretrained COCO weights - works out of the box!

### Train Your First Model

```bash
python scripts/train.py \
    --train-img-dir data/VisDrone2019-DET-train/images \
    --train-ann-dir data/VisDrone2019-DET-train/annotations \
    --val-img-dir data/VisDrone2019-DET-val/images \
    --val-ann-dir data/VisDrone2019-DET-val/annotations \
    --model fasterrcnn_mobilenet \
    --epochs 10 \
    --batch-size 4 \
    --output-dir outputs/first_model
```

## Troubleshooting

### Issue: "No module named 'visdrone_toolkit'"

**Solution:**

```bash
# Make sure you're in the right directory
cd /path/to/VisDrone-dataset-python-toolkit

# Install in editable mode
pip install -e .
```

### Issue: "CUDA out of memory"

**Solutions:**

1. Reduce batch size: `--batch-size 2` or `--batch-size 1`
2. Use smaller model: `--model fasterrcnn_mobilenet`
3. Enable AMP: `--amp`
4. Close other programs using GPU

### Issue: "torch.cuda.is_available() returns False"

**Solutions:**

1. Check NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA:

   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: ImportError for cv2 or other packages

**Solution:**

```bash
pip install opencv-python matplotlib pillow tqdm
```

### Issue: "Permission denied" on Linux/Mac

**Solution:**

```bash
# Add execute permissions
chmod +x scripts/*.py
```

## Updating

### Update Package

```bash
git pull
pip install -e . --upgrade
```

### Update Dependencies

```bash
pip install -r requirements.txt --upgrade
```

## Uninstallation

```bash
# Deactivate virtualenv
deactivate

# Remove virtualenv
rm -rf venv/

# Remove package
pip uninstall visdrone-toolkit
```

## Docker Alternative (Optional)

If you prefer Docker:

```bash
# Build image
docker build -t visdrone-toolkit .

# Run container
docker run --gpus all -it visdrone-toolkit bash
```

## Next Steps

After installation:

1. ✅ Test webcam demo
2. ✅ Download VisDrone dataset
3. ✅ Train your first model
4. ✅ Read the documentation

See `README.md` for usage examples and `scripts/README.md` for script documentation.

## Getting Help

- **GitHub Issues:** <https://github.com/dronefreak/VisDrone-dataset-python-toolkit/issues>
- **Documentation:** Check README.md and docs/
- **Examples:** See examples/ directory

## System Requirements

### Minimum

- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended

- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 50GB+ disk space (for datasets)

### Optimal

- Python 3.11+
- 32GB+ RAM
- NVIDIA RTX 3090 or better
- SSD storage
