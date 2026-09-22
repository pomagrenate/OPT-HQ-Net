# Two-Stage Self-Supervised and Supervised Micro-Segmentation Framework

Complete implementation of SimMIM pre-training followed by supervised fine-tuning for solar filament segmentation using SegFormer B0 on single-channel grayscale images.

## Architecture Overview

### Stage 1: Self-Supervised SimMIM Pre-training
- **Encoder**: SegFormer B0 adapted for 1-channel grayscale input
- **Masking Strategy**: Random block masking (32x32 patches, 50-60% coverage)
- **Reconstruction Decoder**: Lightweight decoder from stage 4 to full resolution
- **Loss**: L1 reconstruction loss on masked regions only
- **Optimization**: AdamW, Cosine Annealing with warmup, AMP

### Stage 2: Supervised Fine-tuning
- **Encoder Initialization**: Load pre-trained weights from Stage 1
- **Decoder**: Multi-scale Feature Pyramid Decoder (4 stages)
- **Output**: Dual-channel logits (mask + skeleton)
- **Loss**: Compound loss (BCE + Dice + clDice + skeleton BCE)
- **Augmentation**: YOLO-style synchronized transforms

## File Structure

```
src/
├── simmim.py              # SimMIM pre-training model
├── segmentation.py       # Supervised segmentation model
├── dataset.py             # Dataset with YOLO-style augmentation
└── losses.py              # Compound loss (BCE + Dice + clDice + skeleton)

train_simmim.py            # SimMIM pre-training script
train_segmentation.py      # Supervised fine-tuning script
```

## Stage 1: SimMIM Pre-training

### Training Command
```bash
python train_simmim.py \
    --data_root E:\GithubProjects\solar_filament_seg\MAGFiLO_1.0_Kaggle_2026\train \
    --backbone nvidia/mit-b0 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1.5e-4 \
    --mask_ratio 0.5 \
    --warmup_epochs 5 \
    --save_dir checkpoints/simmim \
    --use_amp \
    --tile_size 512 \
    --stride 512
```

### Key Arguments
- `--mask_ratio`: Masking ratio (default: 0.5 for 50% coverage)
- `--warmup_epochs`: Linear warmup epochs (default: 5)
- `--lr`: Learning rate (default: 1.5e-4)
- `--weight_decay`: Weight decay (default: 0.05)
- `--use_amp`: Enable Automatic Mixed Precision

### Output
- Checkpoints saved to `checkpoints/simmim/`
- Best model: `simmim_best.pt`
- Periodic checkpoints: `simmim_epoch_{N}.pt`

## Stage 2: Supervised Fine-tuning

### Training Command (with SimMIM weights)
```bash
python train_segmentation.py \
    --data_root E:\GithubProjects\solar_filament_seg\MAGFiLO_1.0_Kaggle_2026\train \
    --backbone nvidia/mit-b0 \
    --simmim_checkpoint checkpoints/simmim/simmim_best.pt \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --fg_ratio 0.70 \
    --bnd_ratio 0.20 \
    --save_dir checkpoints/segmentation \
    --use_amp \
    --tile_size 512 \
    --stride 512
```

### Training Command (ImageNet pretrained only)
```bash
python train_segmentation.py \
    --data_root E:\GithubProjects\solar_filament_seg\MAGFiLO_1.0_Kaggle_2026\train \
    --backbone nvidia/mit-b0 \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --fg_ratio 0.70 \
    --bnd_ratio 0.20 \
    --save_dir checkpoints/segmentation \
    --use_amp \
    --tile_size 512 \
    --stride 512
```

### Key Arguments
- `--simmim_checkpoint`: Path to SimMIM pre-trained encoder
- `--fg_ratio`: Foreground tile ratio (default: 0.70)
- `--bnd_ratio`: Boundary tile ratio (default: 0.20)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.01)

### Output
- Checkpoints saved to `checkpoints/segmentation/`
- Best model: `seg_best.pt`
- Periodic checkpoints: `seg_epoch_{N}.pt`

## Model Details

### SimMIM Model (`src/simmim.py`)
- **Input**: (B, 1, 512, 512) - grayscale
- **Encoder**: SegFormer B0 with 1-channel adaptation
- **Masking**: Random block masking (32x32 patches)
- **Decoder**: Progressive upsampling (stride 32 → 16 → 8 → 4 → 2 → 1)
- **Output**: (B, 1, 512, 512) - reconstructed grayscale
- **Parameters**: ~5.2M (encoder + decoder)

### Segmentation Model (`src/segmentation.py`)
- **Input**: (B, 1, 512, 512) - grayscale
- **Encoder**: SegFormer B0 (can load SimMIM weights)
- **Decoder**: Feature Pyramid (4 stages)
- **Output**: (B, 2, 512, 512) - mask + skeleton logits
- **Parameters**: ~4.7M (encoder) + ~0.5M (decoder)

## Augmentation Strategy

YOLO-style synchronized augmentation applied to image, mask, and skeleton:
- **Geometric**: RandomRotate90, HorizontalFlip, VerticalFlip, Affine
- **Multi-scale**: RandomResizedCrop
- **Physics-aware**: ElasticTransform (plasma simulation)
- **Photometric**: RandomBrightnessContrast, GaussNoise (skipped for grayscale)

## Loss Formulation

### SimMIM Loss
```python
loss = L1(reconstructed * mask, original * mask)
```
Computed only on masked regions to force reconstruction learning.

### Segmentation Loss
```python
loss = loss_bce + loss_dice + loss_cldice + loss_skel
```
- **BCE**: Binary cross-entropy (handles class imbalance)
- **Dice**: Dice loss (optimizes overlap)
- **clDice**: Soft skeleton topology loss (GPU-based)
- **Skeleton BCE**: Skeleton-specific BCE

## Weight Transfer

The segmentation model can load SimMIM pre-trained encoder weights:

```python
model = SolarFilamentSegmentation(
    backbone_name="nvidia/mit-b0",
    in_channels=1,
    simmim_checkpoint="checkpoints/simmim/simmim_best.pt"
)
```

The `load_simmim_weights()` method handles:
- Matching encoder parameters
- Skipping mismatched keys (decoder heads)
- Preserving 1-channel adaptation

## Training Tips

### SimMIM Pre-training
- Use larger batch size (16-32) for stable masking
- Mask ratio 0.5-0.6 works well
- 100 epochs typically sufficient
- AMP reduces memory by ~40%

### Supervised Fine-tuning
- Lower learning rate (1e-4) with SimMIM weights
- Use balanced tile sampling (fg_ratio=0.70, bnd_ratio=0.20)
- Overlapping tiles for validation (stride=256)
- 50 epochs typically sufficient

## Performance

### Expected Training Time (2x T4 GPUs)
- **SimMIM**: ~15-20 minutes/epoch (batch_size=16)
- **Segmentation**: ~2-3 minutes/epoch (batch_size=8)

### Expected Results
- **SimMIM**: L1 loss ~0.05-0.10 after 100 epochs
- **Segmentation**: Dice ~0.85-0.90 after 50 epochs with SimMIM pre-training

## Verification

Test the complete pipeline:
```bash
# Test SimMIM model
python -c "from src.simmim import SimMIMSegFormer; import torch; model = SimMIMSegFormer(); x = torch.randn(2, 1, 512, 512); out = model(x); print('SimMIM output:', out['reconstructed'].shape)"

# Test segmentation model
python -c "from src.segmentation import SolarFilamentSegmentation; import torch; model = SolarFilamentSegmentation(); x = torch.randn(2, 1, 512, 512); out = model(x); print('Segmentation output:', out.shape)"
```

## Advantages of Two-Stage Framework

1. **Self-Supervised Learning**: Learns long-range magnetic topology without annotations
2. **Data Efficiency**: SimMIM can use unlabeled data for pre-training
3. **Better Generalization**: Pre-trained encoder captures plasma continuity
4. **Faster Convergence**: Supervised fine-tuning converges faster
5. **Improved Accuracy**: Expected 2-5% Dice improvement over ImageNet-only pre-training

## Notes

- All models use 1-channel grayscale input (no pseudo-RGB expansion)
- SegFormer B0 provides efficient self-attention for long-range dependencies
- SimMIM pre-training is optional but recommended for best performance
- The framework is modular - can use ImageNet pre-training directly if SimMIM is skipped
