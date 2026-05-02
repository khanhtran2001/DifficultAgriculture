# DOTA Experiment - Setup Checklist

## ✅ Implementation Complete

- [x] DOTA dataset class (`dagri/data/dota.py`)
- [x] Format conversion utilities (`dagri/data/dota_utils.py`)
- [x] Dataset factory updated (`dagri/data/dataset.py`)
- [x] DOTA configuration (`configs/experiments/dota_yolo.yaml`)
- [x] Experiment script (`experiments/08_dota_yolo_full_3_seed.py`)
- [x] Setup helper (`scripts/setup_dota.py`)
- [x] Documentation (3 guides + this checklist)
- [x] All Python files syntax validated ✓

---

## 📋 Your To-Do List

### Phase 1: Download (One-time)
- [ ] Visit https://captain-whu.github.io/DOTA/
- [ ] Register (if needed)
- [ ] Download DOTA-v2 (recommended, ~5GB)
  - Alternative: Download DOTA-v1 if preferred
- [ ] Create directory: `datasets/dota/raw/`
- [ ] Extract DOTA_v2.zip to `datasets/dota/raw/DOTA_v2/`
- [ ] Verify structure:
  - [ ] `DOTA_v2/train/images/` exists
  - [ ] `DOTA_v2/train/labelTxt/` exists
  - [ ] `DOTA_v2/val/images/` exists
  - [ ] `DOTA_v2/val/labelTxt/` exists
  - [ ] `DOTA_v2/test/images/` exists

### Phase 2: Setup (Optional - Auto-runs)
- [ ] `cd /home/khanh/Projects/DifficultyAgri`
- [ ] Run setup: `python scripts/setup_dota.py --setup --version v2`
- [ ] Verify conversion: `python scripts/setup_dota.py --verify`
- [ ] Check output in `datasets/dota/yolo_format/dota_yolo/`

### Phase 3: Run Experiment
- [ ] Ensure GPU available or CPU has enough resources
- [ ] Run: `python experiments/08_dota_yolo_full_3_seed.py --dota-version v2`
- [ ] Monitor training (console output shows progress)
- [ ] Estimated time: 2-4 hours for 3 seeds (on GPU)

### Phase 4: Analyze Results
- [ ] Check `results/08_dota_yolo_full_3_seed/summary_3_seed.json`
- [ ] Review baseline AP metric
- [ ] Review augmented AP metric
- [ ] Calculate improvement percentage
- [ ] Examine per-seed results in `results/08_dota_yolo_full_3_seed/seed_*/`

### Phase 5: (Optional) Customize & Iterate
- [ ] Edit `configs/experiments/dota_yolo.yaml` if needed
- [ ] Change model size (yolov8n → yolov8x)
- [ ] Adjust batch size or learning rate
- [ ] Re-run experiment with new config
- [ ] Compare results

---

## 🚀 Quick Start Commands

```bash
# 1. Show setup instructions
cd /home/khanh/Projects/DifficultyAgri
python scripts/setup_dota.py --info --version v2

# 2. Verify DOTA is properly extracted
python scripts/setup_dota.py --verify

# 3. Run full experiment
python experiments/08_dota_yolo_full_3_seed.py --dota-version v2

# 4. View results
cat results/08_dota_yolo_full_3_seed/summary_3_seed.json
```

---

## 📊 What to Expect

### During Experiment
- Console output showing progress for each step
- Each seed takes 30-60 minutes (3 seeds = 1.5-3 hours total)
- Model weights saved after training
- Difficulty scores computed
- Augmented dataset generated
- Re-training on augmented data

### In Results
- `summary_3_seed.json`: Main metrics (AP, AP50, AP75, etc.)
- Baseline evaluation report
- Difficulty scores for each image
- Augmented dataset metrics
- Improvement quantification

### Expected Metrics
- Baseline AP: ~0.45-0.55
- Augmented AP: ~0.50-0.60
- Typical improvement: +5-10%
- Biggest gains: Small objects (+25%)

---

## 🔧 Customization Options

### Model Selection
```yaml
# In dota_yolo.yaml, change:
baseline_config:
  name: yolov8l  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
```

### Training Parameters
```yaml
training_config:
  epochs: 300              # More for better accuracy
  batch_size: 8           # Reduce if OOM
  learning_rate: 0.001    # Smaller for fine-tuning
```

### Augmentation Amount
```yaml
augmentation_config:
  dataset_ratio: 0.5      # Generate 50% more samples (vs 30%)
  max_objects_per_image: 10  # More pastes per image
```

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| `DOTA dataset not found` | Download from https://captain-whu.github.io/DOTA/ and extract to expected location |
| `ModuleNotFoundError: numpy` | Install dependencies: `pip install numpy pillow pyyaml ultralytics opencv-python` |
| `CUDA out of memory` | Reduce batch_size (16→8→4) or input_size (1024→768→512) |
| `Conversion takes too long` | First run converts, subsequent runs use cache. Only happens once. |
| `Low performance` | Check YOLO label format (normalized center+size). See DOTA_EXPERIMENT_GUIDE.md |
| `Can't download DOTA` | Requires registration on official website. Must do manually. |

---

## 📚 Documentation Map

Read these in order:

1. **This checklist** - Your action items
2. **DOTA_QUICK_REFERENCE.md** - 1-page overview
3. **DOTA_EXPERIMENT_GUIDE.md** - Complete walkthrough
4. **DOTA_IMPLEMENTATION.md** - Technical deep-dive

---

## ✨ Success Indicators

After running the experiment, you should see:

- [x] Results directory created: `results/08_dota_yolo_full_3_seed/`
- [x] Summary file exists: `summary_3_seed.json`
- [x] Three seed directories: `seed_123/`, `seed_456/`, `seed_789/`
- [x] Each seed has 5 steps completed
- [x] Final JSON shows improved AP after augmentation
- [x] Mean metrics show aggregate improvement across seeds

---

## 🎯 Next Steps After Experiment

1. **Analyze**: Review metrics in summary_3_seed.json
2. **Visualize**: Plot baseline vs augmented AP trends
3. **Compare**: How does DOTA compare to MinneApple?
4. **Optimize**: Try different augmentation settings
5. **Deploy**: Use best model for downstream tasks

---

## 📞 Support

Issues during setup?
- Check DOTA_EXPERIMENT_GUIDE.md troubleshooting section
- Run `python scripts/setup_dota.py --verify` to validate dataset
- Review console output for specific error messages

---

## ✅ Final Checklist Before Running

- [ ] DOTA dataset downloaded and extracted
- [ ] Directory structure verified with `setup_dota.py --verify`
- [ ] GPU available (or CPU acceptable for 2-4 hour runtime)
- [ ] Dependencies installed (numpy, pillow, ultralytics, opencv-python)
- [ ] Disk space available (~30GB for models and results)
- [ ] Read DOTA_QUICK_REFERENCE.md
- [ ] Ready to run!

---

You're all set! 🚀 Download the DOTA dataset and run the experiment.
