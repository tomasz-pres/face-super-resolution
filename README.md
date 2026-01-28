# Face Image Super-Resolution using FFHQ Dataset

A deep learning project for 4x face image super-resolution, upscaling 64x64 images to 256x256 with enhanced detail and perceptual quality.

## Project Overview

This project implements Single Image Super-Resolution (SISR) specialized for human face images using the FFHQ dataset. Three model architectures are compared:

1. **Baseline ESRGAN** - Pre-trained model for benchmark comparison
2. **Transfer Learning** - Fine-tuned ESRGAN backbone with face-specific head
3. **Custom FaceEnhanceNet** - Novel architecture with Channel Attention mechanism

### Target Metrics
- PSNR: >= 28 dB
- SSIM: >= 0.85
- LPIPS: <= 0.15
- Inference: < 100ms on GPU

## Project Structure

```
face-super-resolution/
├── configs/              # Training configurations
├── data/                 # Dataset (DVC tracked)
│   ├── raw/              # Original FFHQ images
│   └── processed/        # LR-HR pairs
├── src/                  # Source code
│   ├── data/             # Dataset classes
│   ├── models/           # Model architectures
│   ├── losses/           # Loss functions
│   ├── training/         # Training utilities
│   ├── evaluation/       # Metrics and visualization
│   └── utils/            # Helper functions
├── scripts/              # Training and evaluation scripts
├── notebooks/            # Jupyter notebooks
├── app/                  # Gradio demo application
├── checkpoints/          # Model checkpoints
├── outputs/              # Generated outputs
└── tests/                # Unit tests
```

## Requirements

### Hardware
- NVIDIA GPU with >= 8GB VRAM (recommended: 16GB)
- 32GB RAM
- 100GB disk space for dataset

### Software
- Python >= 3.10
- CUDA >= 11.8
- Docker (optional)

## Installation

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/face-super-resolution.git
cd face-super-resolution

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker

```bash
# Build image
docker build -t face-sr .

# Run training
docker-compose up training

# Run demo
docker-compose up demo
```

## Dataset Preparation

### Download FFHQ

```bash
# Option 1: Kaggle
kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq
unzip flickrfaceshq-dataset-ffhq.zip -d data/raw/

# Option 2: Official script
git clone https://github.com/NVlabs/ffhq-dataset.git
cd ffhq-dataset && python download_ffhq.py --images
```

### Prepare LR-HR Pairs

```bash
python src/data/prepare_data.py \
    --input data/raw/ffhq \
    --output data/processed \
    --hr-size 256 \
    --lr-size 64 \
    --degradation bicubic
```

## Training

### Train Custom Model

```bash
python scripts/train.py --config configs/config.yaml
```

### Train with Specific Model

```bash
# Custom FaceEnhanceNet
python scripts/train.py --config configs/config.yaml --model custom

# Transfer Learning
python scripts/train.py --config configs/config.yaml --model transfer
```

### Monitor Training

Training progress is logged to Weights & Biases. Set your API key:

```bash
export WANDB_API_KEY=your_api_key
wandb login
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --test-dir data/processed/test
```

## Demo

### Gradio Interface

```bash
python app/gradio_app.py
```

Access the demo at `http://localhost:7860`

### Single Image Inference

```bash
python scripts/inference.py \
    --input path/to/image.jpg \
    --output path/to/output.jpg \
    --checkpoint checkpoints/best_model.pth
```

## Model Architectures

### FaceEnhanceNet (Custom)

- 8 Residual Channel Attention Blocks (RCAB)
- SE-style channel attention mechanism
- PixelShuffle upsampling (2x 2x stages)
- ~2.1M parameters

### Loss Functions

1. **L1 Loss** - Pixel-wise reconstruction
2. **Perceptual Loss** - VGG19 feature matching
3. **SSIM Loss** - Structural similarity
4. **Combined Loss** - Weighted combination

## Results

| Model | PSNR (dB) | SSIM | LPIPS | Params | Inference |
|-------|-----------|------|-------|--------|-----------|
| Bicubic | ~24.0 | ~0.70 | ~0.40 | 0 | - |
| ESRGAN (baseline) | ~27.5 | ~0.82 | ~0.18 | 16.7M | ~50ms |
| Transfer Learning | ~28.0 | ~0.84 | ~0.16 | 12.5M | ~40ms |
| FaceEnhanceNet | ~28.5 | ~0.86 | ~0.14 | 2.1M | ~15ms |

## DVC Usage

```bash
# Initialize DVC
dvc init

# Pull data
dvc pull

# Reproduce pipeline
dvc repro
```

## Weights & Biases Integration

All experiments are tracked with W&B:
- Loss curves and metrics
- Sample predictions
- Hyperparameter logging
- Model checkpoints

## License

MIT License

## Acknowledgments

- FFHQ Dataset: [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)
- ESRGAN: [xinntao/ESRGAN](https://github.com/xinntao/ESRGAN)
- BasicSR: [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR)
