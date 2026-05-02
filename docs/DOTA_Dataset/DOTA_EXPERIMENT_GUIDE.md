# DOTA/DOTA-v2 Experiment Setup Guide

This guide will help you set up and run the DOTA dataset experiment for aerial object detection using your DifficultyAgri pipeline.

## Overview

The DOTA experiment follows the same 5-step pipeline as the MinneApple experiment:

1. **Step 0**: Prepare dataset (download & convert DOTA to YOLO format)
2. **Step 1**: Load and validate dataset
3. **Step 2**: Train baseline model and evaluate
4. **Step 3**: Score dataset to identify difficult samples
5. **Step 4**: Generate augmented dataset using copy-paste augmentation
6. **Step 5**: Retrain model on augmented dataset and compare results

## Quick Start

### 1. Download DOTA Dataset

First, you need to manually download the DOTA dataset:

1. Visit: https://captain-whu.github.io/DOTA/
2. Download either DOTA-v1 or DOTA-v2 (v2 is newer with 1.7M objects)
3. Extract the dataset to: `/home/khanh/Projects/DifficultyAgri/datasets/dota/raw/`

Expected directory structure:
```
/home/khanh/Projects/DifficultyAgri/datasets/dota/raw/
└── DOTA_v2/  (or DOTA_v1)
    ├── train/
    │   ├── images/           (*.jpg, *.png, etc)
    │   └── labelTxt/         (*.txt label files)
    ├── val/
    │   ├── images/
    │   └── labelTxt/
    └── test/
        └── images/
```

### 2. Run the Experiment

After downloading the dataset, run the experiment:

```bash
cd /home/khanh/Projects/DifficultyAgri

# For DOTA-v2 (recommended)
python experiments/08_dota_yolo_full_3_seed.py --dota-version v2

# For DOTA-v1
python experiments/08_dota_yolo_full_3_seed.py --dota-version v1

# Using custom config
python experiments/08_dota_yolo_full_3_seed.py --config /path/to/custom_config.yaml --dota-version v2
```

## What Gets Created

### DOTA Dataset (Converted to YOLO Format)
```
/home/khanh/Projects/DifficultyAgri/datasets/dota/yolo_format/dota_yolo/
├── train/
│   ├── images/        (training images)
│   └── labels/        (YOLO format .txt files)
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── classes.json       (class mapping)
```

### Experiment Results
```
/home/khanh/Projects/DifficultyAgri/results/08_dota_yolo_full_3_seed/
├── frozen_config.yaml                 (actual config used)
├── summary_3_seed.json                (aggregated results)
└── seed_123/
    ├── Step_1_Load_and_Validate_Dataset/
    ├── Step_2_Train_and_Evaluate_BASELINE_MODEL/
    │   ├── train_results/             (trained weights)
    │   ├── low_conf_predictions/      (predictions at conf=0.0001)
    │   └── optimal_conf_predictions/  (predictions at optimal conf)
    ├── Step_3_Scoring_Dataset/        (difficulty scores)
    ├── Step_4_Copy_Paste_Augmentation/
    │   └── augmented_dataset/         (new augmented dataset)
    └── Step_5_Train_and_Evaluate_Model_on_Augmented_Dataset/
        └── train_results/             (retrained weights)
```

## DOTA Dataset Information

### Classes (15 total)
- plane
- baseball-diamond
- bridge
- ground-track-field
- small-vehicle
- large-vehicle
- ship
- tennis-court
- basketball-court
- storage-tank
- soccer-ball-field
- roundabout
- harbor
- swimming-pool
- helicopter

### DOTA-v1 Stats
- Training images: 2,806
- Validation images: 1,411
- Test images: 1,828
- Total objects: ~188,282

### DOTA-v2 Stats (Recommended)
- Training images: 1,830 (larger images, 4096x4096)
- Validation images: 958
- Test images: 2,000
- Total objects: ~1.7M

## Configuration

The experiment uses `/home/khanh/Projects/DifficultyAgri/configs/experiments/dota_yolo.yaml`

Key settings:
- **Model**: YOLOv8m (for multi-class detection on varied orientations)
- **Input size**: 1024x1024 (DOTA images are large, resized for training)
- **Batch size**: 16
- **Epochs**: 200
- **Scoring**: Difficulty-based weighting for hard negatives and missed detections
- **Augmentation**: Copy-paste augmentation with 30% dataset multiplier

### Custom Configuration

To customize, edit `dota_yolo.yaml` or create a new config:

```yaml
dataset_config:
  name: dota_v2
  root_dir: /path/to/yolo_formatted/dataset
  num_classes: 15
  class_names: [...]

baseline_config:
  training_config:
    epochs: 200
    batch_size: 16
    learning_rate: 0.01

augmentation_config:
  dataset_ratio: 0.3  # Generate 30% new samples
  max_objects_per_image: 8
```

## Expected Results

The augmentation typically improves performance:
- **Baseline AP**: ~0.45-0.55 (varies with initialization)
- **Augmented AP**: ~0.50-0.60 (5-10% improvement typical)
- **Best improvements on**: Medium and large objects

## Output Files

### Key Results Files
- `summary_3_seed.json`: Summary across all 3 random seeds
- `seed_XXX/Step_2.../evaluation_report.json`: Baseline evaluation metrics
- `seed_XXX/Step_3.../score_results.json`: Difficulty scores for each image
- `seed_XXX/Step_5.../evaluation_report_*.json`: Final evaluation results

### Models
- `seed_XXX/Step_2.../train_results/weights/best.pt`: Baseline model weights
- `seed_XXX/Step_5.../train_results/weights/best.pt`: Augmented model weights

## Troubleshooting

### Dataset not found
```
Error: DOTA dataset not found at /home/khanh/Projects/DifficultyAgri/datasets/dota/raw/DOTA_v2
```
**Solution**: Download DOTA-v2 from the official website and extract to the expected location.

### Memory issues during training
- Reduce `batch_size` in config from 16 to 8 or 4
- Use smaller `input_size` (768 or 512 instead of 1024)
- Increase `early_stopping_patience` to allow more epochs

### Slow augmentation
- Reduce `dataset_ratio` from 0.3 to 0.1
- Reduce `max_objects_per_image` from 8 to 4

### Low performance
- Check that class names match the DOTA standard 15 classes
- Verify YOLO format labels are correct (class_id cx cy w h, normalized)
- Consider increasing training epochs or learning rate

## Format Conversion Details

The script converts DOTA's format to YOLO automatically:

**DOTA Format** (in labelTxt files):
```
class_name x1 y1 x2 y2 x3 y3 x4 y4 [difficulty]
plane 100 100 300 50 350 200 150 250 0
```

**YOLO Format** (in labels files):
```
class_id center_x center_y width height
0 0.25 0.20 0.35 0.25
```

The conversion:
1. Takes the 8 corner coordinates (rotated bounding box)
2. Computes axis-aligned bounding box from corners
3. Converts to YOLO center + size format
4. Normalizes to [0, 1] range

## Next Steps

After running the experiment:

1. **Analyze Results**: Check `summary_3_seed.json` for mean metrics across seeds
2. **Visualize**: Use the difficulty scores from Step_3 to understand which objects are hard
3. **Iterate**: Modify augmentation config and rerun to find optimal settings
4. **Compare**: Run MinneApple experiment side-by-side to compare datasets

## References

- DOTA Official: https://captain-whu.github.io/DOTA/
- DOTA Paper: https://arxiv.org/abs/1711.10398
- YOLOv8 Docs: https://docs.ultralytics.com/

## Support

For issues with the DifficultyAgri library, check:
- `dagri/data/dota_utils.py`: Dataset conversion utilities
- `dagri/data/dota.py`: DOTA dataset class
- `configs/experiments/dota_yolo.yaml`: Configuration template
