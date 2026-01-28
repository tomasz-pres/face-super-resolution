# Face Super-Resolution with Deep Learning

A deep learning project for 4x face image super-resolution (64×64 → 256×256) with focus on **perceptual quality and photorealism**. Uses a custom FaceEnhanceNet architecture with Channel Attention and three-stage training strategy combining PSNR optimization with GAN fine-tuning.

## Highlights

- **5x better perceptual quality** than bicubic baseline (LPIPS: 0.0695 vs 0.37)
- Custom architecture with Channel Attention mechanism
- Three-stage progressive training (PSNR → SSIM → GAN)
- Trained on 70,000 high-quality face images from FFHQ dataset
- Interactive Gradio demo with live metrics
- Docker deployment ready

## Model Architecture

**FaceEnhanceNet** - Custom architecture optimized for face super-resolution:

- **6 Residual Groups** with **10 RCAB blocks** each (60 blocks total)
- **Channel Attention** mechanism for adaptive feature recalibration
- **64 feature channels** throughout the network
- **PixelShuffle upsampling** (2×→2× cascade for 4× total)
- Progressive training strategy for optimal perceptual quality

### Key Components

1. **Residual Channel Attention Blocks (RCAB)**
   - Two 3×3 convolutions with ReLU
   - Squeeze-and-Excitation style channel attention (reduction ratio: 4)
   - Residual connections for gradient flow

2. **Upsampling Module**
   - Two-stage PixelShuffle: 64×64 → 128×128 → 256×256
   - Learned upsampling filters for better quality

## Training Strategy

### Three-Stage Progressive Training (170 epochs total)

**Stage 1: PSNR Pre-training (Epochs 0-99)**
- Focus: Pixel-level reconstruction accuracy
- Loss: L1 (1.0) + Perceptual (1.0)
- Learning rate: 1e-4 with Cosine Annealing
- Result: Good PSNR, but outputs lack fine details

**Stage 2: SSIM Fine-tuning (Epochs 100-149)**
- Focus: Structural similarity
- Loss: L1 (1.0) + Perceptual (0.5) + SSIM (0.2)
- Learning rate: 1e-5
- Result: Did not bring significant improvements

**Stage 3: GAN Fine-tuning (Epochs 150-169)**
- Focus: Perceptual quality and photorealism
- Loss: L1 (0.01) + Perceptual (1.0) + Adversarial (0.005)
- Learning rate: 1e-4
- Result: **Major perceptual quality improvements (LPIPS: 0.0695)**

### The Perception-Distortion Trade-off

Traditional metrics (PSNR, SSIM) measure pixel-level accuracy but don't correlate well with human perception. Our GAN-trained model demonstrates this trade-off:
- Similar PSNR to baselines (~26 dB)
- **5× better LPIPS** (perceptual similarity)
- Sharper details, realistic textures, better facial features

## Dataset

**FFHQ (Flickr-Faces-HQ)** - NVIDIA 256px version
- 70,000 high-quality aligned face images (256×256)
- Split: 60,000 train / 5,000 val / 5,000 test
- **On-the-fly LR generation** using bicubic downsampling
- Data augmentation: Horizontal flip (50%)

On-the-fly generation ensures the model never sees the exact same LR-HR pair twice, improving generalization.

## Results

Evaluation on **4,970 test images**:

| Method | PSNR (dB) | SSIM | LPIPS | Notes |
|--------|-----------|------|-------|-------|
| **Baseline Methods** |||||
| Bilinear | 26.43 | 0.7843 | 0.3407 | - |
| Bicubic | 26.31 | 0.7861 | 0.3716 | - |
| Lanczos4 | 26.10 | 0.7754 | 0.3883 | - |
| **Deep Learning Models** |||||
| Transfer Learning | 26.97 | 0.7940 | 0.1081 | Beats baselines on all metrics |
| **Custom (GAN)** | **26.39** | **0.7734** | **0.0695** | **5× better perceptual quality** |

**Key Finding:** Despite similar PSNR to baselines, my GAN model achieves dramatically better perceptual quality (LPIPS 0.0695 vs ~0.37 for baselines), producing photorealistic results with sharp details and realistic textures.

## Installation

### Prerequisites

- Python >= 3.10
- CUDA >= 11.8 (for GPU training/inference)
- NVIDIA GPU with >= 8GB VRAM (recommended)

### Local Setup

```bash
# Clone repository
git clone https://github.com/tomasz-pres/face-super-resolution.git
cd face-super-resolution

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
# Build and run demo
docker-compose up demo

# Access at http://localhost:7860
```

## Usage

### Interactive Demo

```bash
python app/demo.py
```

Open http://localhost:7860 in your browser. Features:
- Upload any face image or select from validation samples
- Real-time super-resolution processing
- Side-by-side comparison with Bicubic and Lanczos4
- Live PSNR, SSIM, and LPIPS metrics

### Training

Reproduce the three-stage training:

```bash
# Stage 1: PSNR Pre-training (100 epochs)
python scripts/train.py --config configs/stages/stage1_psnr_config.yaml

# Stage 2: SSIM Fine-tuning (50 epochs)
python scripts/train.py --config configs/stages/stage2_ssim_config.yaml

# Stage 3: GAN Fine-tuning (20 epochs)
python scripts/train.py --config configs/stages/stage3_gan_config.yaml
```

### Model Comparison

Compare multiple models on test set:

```bash
python scripts/compare_two_models.py \
    --test-dir data/processed/test/HR \
    --save-every 5  # Save every 5th comparison image
```

### Visualize Training

Generate training stage plots:

```bash
python scripts/plot_training_stages.py
```

## Project Structure

```
face-super-resolution/
├── app/
│   └── demo.py              # Gradio demo application
├── configs/
│   ├── config.yaml          # Main training config
│   └── stages/              # Stage-specific configs
│       ├── stage1_psnr_config.yaml
│       ├── stage2_ssim_config.yaml
│       ├── stage3_gan_config.yaml
│       └── training_pipeline.yaml
├── src/
│   ├── models/              # FaceEnhanceNet architecture
│   ├── losses/              # Loss functions (L1, Perceptual, SSIM, GAN)
│   ├── training/            # Training utilities
│   ├── evaluation/          # Metrics (PSNR, SSIM, LPIPS)
│   └── data/                # Dataset classes
├── scripts/
│   ├── train.py             # Training script
│   ├── compare_two_models.py # Model comparison
│   ├── plot_training_stages.py # Training visualization
│   └── extract_configs.py   # Config extraction from checkpoints
├── reports/
│   └── figures/             # Training plots and visualizations
├── outputs/
│   └── compare_models/
│       └── samples/         # Comparison images
├── checkpoints/             # Model weights (not in repo)
├── data/                    # Dataset (not in repo)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Configuration

Stage-specific configurations are available in `configs/stages/`:
- Complete hyperparameters for each training stage
- Loss weights and learning rate schedules
- Model architecture parameters
- Fully reproducible training pipeline

## Technical Details

**Training Platform:** RunPod
**Total Training Time:** 170 epochs
**Batch Size:** 48
**Optimizer:** Adam
**Hardware:** NVIDIA GPU with CUDA support

**Loss Functions:**
- L1 Loss: Pixel-wise reconstruction
- Perceptual Loss: VGG19 feature matching
- SSIM Loss: Structural similarity
- Adversarial Loss: GAN training for realism

## Future Improvements

- Support for real-world degradations (noise, blur, compression)
- Higher scale factors (8×, 16×)
- Lightweight model variants for mobile deployment
- Video super-resolution extension

## License

MIT License

## Acknowledgments

- **FFHQ Dataset:** [ffhq-dataset](https://www.kaggle.com/datasets/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px)

- **RCAN Paper:** Zhang et al. "Image Super-Resolution Using Very Deep Residual Channel Attention Networks" (ECCV 2018)
- **ESRGAN Paper:** Wang et al. "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" (ECCV 2018)
- **Perceptual Loss:** Johnson et al. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (ECCV 2016)
