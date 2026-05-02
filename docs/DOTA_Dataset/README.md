# DOTA Dataset Documentation

Complete documentation and guides for setting up and running DOTA/DOTA-v2 experiments with DifficultyAgri.

## 📖 Documentation Files

### Quick Start
- **DOTA_CHECKLIST.md** - Your action items and quick commands
  - Download instructions
  - Setup and verification steps
  - What to expect during and after experiments
  
### Getting Started
- **DOTA_QUICK_REFERENCE.md** - 1-page overview
  - File reference
  - Quick start (3 steps)
  - Key configuration settings
  - Expected results
  - Troubleshooting

### Detailed Guides
- **DOTA_EXPERIMENT_GUIDE.md** - Comprehensive walkthrough
  - Full setup instructions with explanations
  - Configuration options and customization
  - Output file descriptions
  - Format conversion details
  - Support and references

### Technical Reference
- **DOTA_IMPLEMENTATION.md** - Technical deep-dive
  - Architecture and design
  - Integration with DifficultyAgri library
  - Coordinate conversion explanation
  - Troubleshooting for developers
  - Academic references

## 🚀 How to Use

**First time?** Start with DOTA_CHECKLIST.md

**Need quick overview?** Read DOTA_QUICK_REFERENCE.md

**Want detailed explanation?** See DOTA_EXPERIMENT_GUIDE.md

**Developer/Technical?** Check DOTA_IMPLEMENTATION.md

## 📁 Related Files

Scripts for DOTA setup:
- `scripts/download_dota_v1.py` - Download and format DOTA-v1
- `scripts/setup_dota.py` - Setup and verify DOTA dataset
- `scripts/dota_v1_help.py` - Download help and instructions

Experiment code:
- `experiments/08_dota_yolo_full_3_seed.py` - Main experiment pipeline
- `configs/experiments/dota_yolo.yaml` - Configuration for DOTA experiments
- `dagri/data/dota.py` - DOTA dataset class
- `dagri/data/dota_utils.py` - Format conversion utilities

## ✅ What's Included

- ✅ Complete DOTA/DOTA-v2 dataset integration
- ✅ Format conversion (DOTA → YOLO)
- ✅ Download and setup scripts
- ✅ Full experimental pipeline (5 steps)
- ✅ Configuration templates
- ✅ Comprehensive documentation
- ✅ Troubleshooting guides

## 🔗 Quick Links

- **Official DOTA**: https://captain-whu.github.io/DOTA/
- **DOTA Paper (v1)**: https://arxiv.org/abs/1711.10398
- **DOTA Paper (v2)**: https://arxiv.org/abs/2106.11215
- **Ultralytics DOTA-v2**: https://docs.ultralytics.com/datasets/obb/dota-v2/

## 📊 Dataset Info

**DOTA-v2 (Recommended)**
- Training: 1,830 images (4096×4096)
- Validation: 958 images
- Test: 2,000 images
- Classes: 15 (plane, ship, vehicle, building, etc.)
- Total objects: ~1.7 million

**DOTA-v1**
- Training: 2,806 images (1024×1024)
- Validation: 1,411 images
- Test: 1,828 images
- Classes: 15 (same as v2)
- Total objects: ~188,282

## 🎯 Next Steps

1. Read DOTA_CHECKLIST.md
2. Download DOTA dataset from official website
3. Run setup scripts from `scripts/` directory
4. Execute the experiment
5. Analyze results

---

For questions or issues, refer to the troubleshooting sections in the respective documentation files.
