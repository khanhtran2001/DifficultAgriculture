# DOTA Experiment Implementation Summary

## ✓ What Was Created

A complete DOTA/DOTA-v2 (aerial object detection) experiment setup for your DifficultyAgri library. This includes dataset handling, configuration, and a full 3-seed experimental pipeline.

---

## 📁 Files Created/Modified

### 1. Dataset Integration
- **`dagri/data/dota.py`** (NEW)
  - DOTA dataset class implementing the same interface as MinneApple/WheatHead
  - Handles YOLO format directory structure (train/val/test splits with images/labels)
  - Validates dataset integrity

- **`dagri/data/dataset.py`** (MODIFIED)
  - Added support for DOTA and DOTA_v2 dataset names
  - Integrates DotaYoloDetectionDataset into the CustomDataset factory

### 2. Utilities
- **`dagri/data/dota_utils.py`** (NEW)
  - `download_dota_dataset()`: Manages DOTA dataset directory setup
  - `convert_dota_to_yolo()`: Converts DOTA's 8-corner format to YOLO center+size format
  - `dota_to_yolo_bbox()`: Converts bounding box coordinates using numpy
  - `get_dota_class_names()`: Returns standard 15 DOTA classes
  - Helper functions for class mapping and format conversion

### 3. Configuration
- **`configs/experiments/dota_yolo.yaml`** (NEW)
  - Complete experiment configuration with all 15 DOTA classes
  - Uses YOLOv8m model (medium size, good for multi-class detection)
  - Configured for 1024x1024 input (DOTA images are 4096x4096, downsampled)
  - Scoring and augmentation parameters tuned for aerial imagery

### 4. Experiment Script
- **`experiments/08_dota_yolo_full_3_seed.py`** (NEW)
  - Implements full 5-step pipeline:
    - Step 0: Download and format conversion
    - Step 1: Dataset validation
    - Step 2: Baseline training and evaluation
    - Step 3: Difficulty scoring
    - Step 4: Copy-paste augmentation
    - Step 5: Retrain on augmented dataset
  - Runs across 3 random seeds for statistical significance
  - Generates comprehensive results summary with mean metrics

### 5. Setup Script
- **`setup_dota.py`** (NEW)
  - Interactive setup script to help with DOTA dataset preparation
  - Commands:
    - `--info`: Show download instructions
    - `--convert`: Convert existing DOTA to YOLO format
    - `--verify`: Validate YOLO dataset structure
    - `--setup`: Full setup with conversion

### 6. Documentation
- **`DOTA_EXPERIMENT_GUIDE.md`** (NEW)
  - Comprehensive guide with:
    - Quick start instructions
    - Dataset download links
    - Expected directory structure
    - Configuration options
    - Troubleshooting tips
    - Output file descriptions
    - References to papers and official websites

---

## 🎯 DOTA Dataset Information

### 15 Object Classes
```
0. plane
1. baseball-diamond
2. bridge
3. ground-track-field
4. small-vehicle
5. large-vehicle
6. ship
7. tennis-court
8. basketball-court
9. storage-tank
10. soccer-ball-field
11. roundabout
12. harbor
13. swimming-pool
14. helicopter
```

### DOTA-v2 (Recommended)
- Training images: 1,830 (4096×4096 pixels)
- Validation images: 958
- Test images: 2,000
- Total objects: ~1.7 million
- Download: https://captain-whu.github.io/DOTA/

### DOTA-v1
- Training images: 2,806
- Validation images: 1,411
- Test images: 1,828
- Total objects: ~188,282

---

## 🚀 Quick Start

### Step 1: Download DOTA Dataset
```bash
# Visit https://captain-whu.github.io/DOTA/
# Download DOTA-v2 (recommended)
# Extract to: /home/khanh/Projects/DifficultyAgri/datasets/dota/raw/
```

### Step 2: Convert to YOLO Format (Optional - Automatic)
```bash
cd /home/khanh/Projects/DifficultyAgri
python scripts/setup_dota.py --setup --version v2
```

### Step 3: Run Experiment
```bash
cd /home/khanh/Projects/DifficultyAgri
python experiments/08_dota_yolo_full_3_seed.py --dota-version v2
```

The experiment will:
1. Automatically detect and convert DOTA to YOLO format
2. Train baseline model on 3 random seeds
3. Score dataset to find difficult samples
4. Apply copy-paste augmentation
5. Retrain and compare results
6. Save all results with detailed metrics

---

## 📊 Expected Output Structure

```
results/08_dota_yolo_full_3_seed/
├── frozen_config.yaml                    # Config used
├── summary_3_seed.json                   # Aggregate results
└── seed_123/
    ├── Step_1_Load_and_Validate_Dataset/
    │   └── dataset_properties.json
    ├── Step_2_Train_and_Evaluate_BASELINE_MODEL/
    │   ├── train_results/
    │   ├── evaluation_report.json
    │   ├── low_conf_predictions/
    │   └── optimal_conf_predictions/
    ├── Step_3_Scoring_Dataset/
    │   └── score_results.json
    ├── Step_4_Copy_Paste_Augmentation/
    │   └── augmented_dataset/
    │       ├── train/
    │       ├── val/
    │       └── test/
    └── Step_5_Train_and_Evaluate_Model_on_Augmented_Dataset/
        ├── train_results/
        ├── evaluation_report_initial_dataset.json
        └── evaluation_report_new_dataset.json

datasets/dota/
├── raw/
│   └── DOTA_v2/               (Downloaded manually)
│       ├── train/
│       ├── val/
│       └── test/
└── yolo_format/
    └── dota_yolo/             (Auto-converted)
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── classes.json
```

---

## 🔧 Configuration Guide

Edit `configs/experiments/dota_yolo.yaml` to customize:

```yaml
# Change model size
baseline_config:
  name: yolov8l              # options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  
# Change training parameters
  training_config:
    epochs: 200              # More epochs for larger dataset
    batch_size: 16           # Reduce to 8 if OOM
    learning_rate: 0.01      # Tune based on performance

# Change augmentation strategy
augmentation_config:
  dataset_ratio: 0.3         # Generate 30% new samples
  max_objects_per_image: 8   # Max pastes per image
  blending_method: none      # or seamless_clone, alpha, lab_gaussian
```

---

## 📈 Pipeline Overview

The experiment replicates your MinneApple setup but for DOTA:

```
Raw DOTA Dataset
    ↓
[Format Conversion] → YOLO Format
    ↓
[Step 1] Dataset Validation
    ↓
[Step 2] Baseline Training
    ↓
[Step 3] Difficulty Scoring
    ↓
[Step 4] Copy-Paste Augmentation
    ↓
[Step 5] Retrain & Evaluate
    ↓
Results (3 seeds, mean metrics)
```

Each step:
- Uses the same configuration as MinneApple
- Follows YOLO format standards
- Integrates with your existing scoring/augmentation pipeline
- Generates JSON outputs for further analysis

---

## 🔌 Integration Points

### With Your Library
- Uses `dagri.baseline.Baseline` for YOLOv8 training
- Uses `dagri.scoring.Scorer` for difficulty assessment
- Uses `dagri.augmentation.CopyPasteAugmentor` for augmentation
- Uses `dagri.general.ResultManager` for result saving
- Uses `dagri.general.ConfigManager` for configuration loading

### Dataset Format
- Follows standard YOLO format (same as MinneApple)
- Images in `split/images/` (jpg, png, tif supported)
- Labels in `split/labels/` (normalized center + size format)
- One label per image (matched by filename)

---

## ⚙️ Technical Details

### Coordinate Conversion
DOTA uses oriented bounding boxes (8 corners), converted to axis-aligned:
```
DOTA:  x1 y1 x2 y2 x3 y3 x4 y4 (pixel coordinates)
                ↓
YOLO:  class_id center_x center_y width height (normalized)
```

### Format Validation
- Checks all required directories exist
- Verifies image-label matching
- Handles multiple image formats (.jpg, .png, .tif, .webp)
- Auto-creates class mapping JSON

---

## 🐛 Troubleshooting

### Dataset not found
```bash
python scripts/setup_dota.py --info          # Shows instructions
# Download DOTA manually from official website
python scripts/setup_dota.py --convert       # Converts existing DOTA
```

### Import errors
```bash
# Ensure numpy, pillow, pyyaml are installed
pip install numpy pillow pyyaml ultralytics opencv-python
```

### Memory issues
```yaml
# In dota_yolo.yaml, reduce:
batch_size: 8              # from 16
input_size: 768            # from 1024
```

### Slow conversion
- First run converts DOTA to YOLO (~10-20 mins for DOTA-v2)
- Subsequent runs use cached format (immediate start)

---

## 📚 Related Files

Your existing MinneApple setup (for comparison):
- Experiment: `experiments/07_minneapple_yolo_full_3_seed.py`
- Config: `configs/experiments/minneapple_yolo.yaml`
- Dataset: `dagri/data/minneapple.py`

---

## 🎓 References

- DOTA Dataset: https://captain-whu.github.io/DOTA/
- DOTA Paper: https://arxiv.org/abs/1711.10398 (v1)
- DOTA v2 Paper: https://arxiv.org/abs/2106.11215
- YOLOv8 Docs: https://docs.ultralytics.com/
- Your MinneApple experiments for methodology

---

## ✨ Next Steps

1. **Download**: Get DOTA-v2 from official website (requires registration)
2. **Setup**: Run `python scripts/setup_dota.py --setup`
3. **Run**: Execute `python experiments/08_dota_yolo_full_3_seed.py`
4. **Analyze**: Review `results/08_dota_yolo_full_3_seed/summary_3_seed.json`
5. **Visualize**: Use difficulty scores to understand challenging objects

---

All files are ready to use! Just download the DOTA dataset and run the experiment.
