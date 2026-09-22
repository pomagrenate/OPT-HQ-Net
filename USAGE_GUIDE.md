# Solar Filament Segmentation - Usage Guide

This guide shows how to use the updated solar filament segmentation code with the MAGFiLO_1.0_Kaggle_2026 dataset.

## 📁 Dataset Structure

Your MAGFiLO dataset should be organized as follows:

```
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/                    # H-alpha JPEG images
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json  # COCO-format annotations
└── test/
    └── test_images/                     # H-alpha JPEG images for inference
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Using the main interface:

```bash
python main.py train --data_root /path/to/MAGFiLO_1.0_Kaggle_2026
```

For Kaggle (memory-optimized):
```bash
python main.py train --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 --batch_size 2 --tile_size 128
```

Or using the standalone training script:

```bash
python train.py --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 --epochs 50 --batch_size 4
```

### 3. Run Inference

Using the main interface:

```bash
python main.py predict --weights checkpoints/best_model.pt --data_root /path/to/MAGFiLO_1.0_Kaggle_2026
```

Or using the standalone inference script:

```bash
python predict.py --weights checkpoints/best_model.pt --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 --output submission.csv
```

## 📋 Command Options

### Training Commands

#### Basic Training
```bash
python main.py train --data_root /path/to/MAGFiLO_1.0_Kaggle_2026
```

#### Training with Custom Parameters
```bash
python main.py train \
    --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --use_amp \
    --use_ema
```

#### Resume Training
```bash
python main.py train \
    --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
    --resume last
```

#### Resume from Specific Checkpoint
```bash
python main.py train \
    --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
    --resume checkpoints/checkpoint_epoch_25.pt
```

### Inference Commands

#### Basic Inference
```bash
python main.py predict \
    --weights checkpoints/best_model.pt \
    --data_root /path/to/MAGFiLO_1.0_Kaggle_2026
```

#### Inference with Custom Parameters
```bash
python main.py predict \
    --weights checkpoints/best_model.pt \
    --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
    --threshold 0.5 \
    --min_area 30 \
    --output my_submission.csv
```

## 🔧 Parameter Descriptions

### Training Parameters

- `--data_root`: Path to training data directory (required)
- `--batch_size`: Batch size for training (default: 2, use 2 for Kaggle)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--tile_size`: Size of training patches (default: 128, use 128 for Kaggle)
- `--overlap`: Overlap fraction for tiling (default: 0.25)
- `--use_amp`: Enable automatic mixed precision training
- `--use_ema`: Enable exponential moving average of model weights
- `--checkpoint_dir`: Directory to save checkpoints (default: checkpoints)
- `--resume`: Resume from checkpoint (path or "last")
- `--device`: Device to use (cuda/cpu, default: cuda)

### Inference Parameters

- `--weights`: Path to model weights checkpoint (required)
- `--data_root`: Path to test data directory (required)
- `--threshold`: Probability threshold for binary mask (default: 0.5)
- `--min_area`: Minimum area for connected components (default: 30)
- `--tile_size`: Tile size for inference (default: 256)
- `--overlap`: Overlap fraction for tiling (default: 0.25)
- `--output`: Output CSV file path (default: submission.csv)
- `--device`: Device to use (cuda/cpu, default: cuda)

## 📦 Checkpoints

Training creates the following checkpoints in the `checkpoints/` directory:

- `best_model.pt`: Best model based on validation loss
- `last.pt`: Most recent checkpoint
- `checkpoint_epoch_XX.pt`: Periodic checkpoints every N epochs

## 🎯 Example Workflow

### Complete Training and Inference Workflow

```bash
# 1. Train the model (Kaggle-optimized)
python main.py train \
    --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
    --epochs 50 \
    --batch_size 2 \
    --tile_size 128 \
    --use_amp \
    --use_ema

# 2. Run inference on test data
python main.py predict \
    --weights checkpoints/best_model.pt \
    --data_root /path/to/MAGFiLO_1.0_Kaggle_2026 \
    --output submission.csv

# 3. Check the submission file
# The submission.csv file will contain filament_id and segmentation_rle columns
```

## 🔍 Module Overview

### Core Modules

- `model.py`: MicroFilNet architecture
- `losses.py`: Loss functions (BCE, Dice, clDice, Boundary)
- `preprocessing.py`: Image preprocessing pipeline
- `inference.py`: Tiled inference and postprocessing
- `dataset.py`: Dataset loading and data augmentation
- `utils.py`: RLE encoding/decoding, PQ metric, checkpoint management

### Main Scripts

- `main.py`: Main interface for training and inference
- `train.py`: Standalone training script
- `predict.py`: Standalone inference script

## 💡 Tips

1. **Use AMP for faster training**: Enable `--use_amp` for automatic mixed precision training
2. **Use EMA for better inference**: Enable `--use_ema` for more stable model weights
3. **Adjust threshold**: The default threshold is 0.5, but you may need to tune this for your data
4. **Monitor training**: The training script prints loss components every 10 batches
5. **Resume training**: Use `--resume last` to continue from the most recent checkpoint

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch_size` or `--tile_size`
2. **Slow data loading**: Enable `--use_cache` if you have preprocessed .npy files
3. **Poor results**: Try increasing `--epochs`, adjusting `--lr`, or enabling `--use_ema`

### Data Format Issues

The code supports multiple image formats:
- FITS files (requires astropy)
- PNG/JPEG images
- Preprocessed .npy arrays

If you encounter FITS loading issues, ensure astropy is installed:
```bash
pip install astropy
```

## 📊 Output Format

### Training Output

The training script outputs:
- Loss values (total, BCE, Dice, clDice, Boundary)
- Progress updates every 10 batches
- Checkpoint saving messages

### Inference Output

The inference script outputs:
- Progress for each image
- Number of filaments detected per image
- Final summary with total filaments detected
- `submission.csv` file in Kaggle format

## 🎓 Advanced Usage

### Custom Data Preprocessing

If you want to preprocess your data to .npy format for faster loading:

```python
from preprocessing import preprocess_observation
import numpy as np
from pathlib import Path

# Preprocess and save as .npy
data_path = Path("/path/to/train/images")
output_path = Path("/path/to/cache/images")
output_path.mkdir(parents=True, exist_ok=True)

for img_file in data_path.glob("*.fits"):
    raw = load_fits_as_array(img_file)
    processed = preprocess_observation(raw)
    np.save(output_path / f"{img_file.stem}.npy", processed.image)
```

### Custom Loss Weights

Modify the loss weights in `losses.py`:

```python
criterion = MicroFilNetLoss(
    w_bce=1.0,      # Binary cross-entropy weight
    w_dice=1.0,     # Dice loss weight
    w_cldice_target=0.5,  # clDice weight
    w_boundary=0.3,  # Boundary loss weight
    cldice_warmup_epochs=10  # clDice warmup epochs
)
```

## 📝 Citation

If you use this code, please cite the MAGFiLO dataset:

```bibtex
@article{magfilo2024,
    title={Manually Annotated GONG Filaments from H-Alpha Observations (MAGFiLO)},
    journal={Scientific Data},
    year={2024},
    doi={10.1038/s41597-024-03876-y}
}
```