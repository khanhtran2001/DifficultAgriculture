#!/usr/bin/env python3
"""
DOTA-v1 Manual Download and Setup Guide
Since automated download doesn't work, follow these manual steps.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_manual_download_guide():
    """Print step-by-step manual download instructions."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              DOTA-v1 MANUAL DOWNLOAD AND SETUP GUIDE                       ║
╚════════════════════════════════════════════════════════════════════════════╝

⚠️  Automated download doesn't work because DOTA requires manual registration.

════════════════════════════════════════════════════════════════════════════════

STEP 1: DOWNLOAD DOTA-v1 (Manual)
────────────────────────────────────

1. Open your browser and go to:
   https://captain-whu.github.io/DOTA/

2. Look for the "Download" section

3. You'll need to fill out a form to request access (REQUIRED)
   - Fill in your information
   - Agree to terms of service
   - Submit request

4. Download these two files (each ~4GB):
   • DOTA_v1_split_1of2.zip
   • DOTA_v1_split_2of2.zip

   ⏱️  Each file takes 20-30 minutes to download (4GB each)
   💾 Make sure you have at least 10GB free disk space

════════════════════════════════════════════════════════════════════════════════

STEP 2: PLACE FILES IN CORRECT LOCATION
────────────────────────────────────────

After downloading, move both ZIP files to:

  {}/datasets/dota/raw/

Directory structure should look like:
  datasets/dota/raw/
  ├── DOTA_v1_split_1of2.zip    (4GB)
  └── DOTA_v1_split_2of2.zip    (4GB)

════════════════════════════════════════════════════════════════════════════════

STEP 3: EXTRACT AND CONVERT
────────────────────────────

Run the conversion script:

  cd {}
  python scripts/setup_dota.py --setup --version v1

This will:
  ✓ Extract both ZIP files
  ✓ Organize into DOTA-v1 standard structure
  ✓ Convert to YOLO format automatically
  ✓ Validate the dataset

Expected output directory structure after conversion:
  datasets/dota/yolo_format/dota_yolo/
  ├── train/
  │   ├── images/        (2,806 images)
  │   └── labels/        (YOLO format)
  ├── val/
  │   ├── images/        (1,411 images)
  │   └── labels/
  └── test/
      ├── images/        (1,828 images)
      └── labels/

════════════════════════════════════════════════════════════════════════════════

STEP 4: VERIFY SETUP
────────────────────

After extraction/conversion, verify everything is correct:

  python scripts/setup_dota.py --verify --version v1

You should see:
  ✓ Dataset structure verified!
    Train: 2806 images
    Val:   1411 images
    Test:  1828 images

════════════════════════════════════════════════════════════════════════════════

STEP 5: RUN EXPERIMENT
──────────────────────

Once verified, run the full 5-step experiment pipeline:

  python experiments/08_dota_yolo_full_3_seed.py --dota-version v1

This will:
  1. Load and validate dataset
  2. Train baseline model (3 random seeds)
  3. Score dataset for difficult samples
  4. Generate augmented dataset
  5. Retrain and compare results

Expected duration: 2-4 hours (with GPU)

════════════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING
───────────────

❌ "File is not a zip file"
   → ZIP files weren't downloaded properly
   → Download manually from https://captain-whu.github.io/DOTA/
   → Check file size is ~4GB each

❌ "Permission denied" during extraction
   → Run with correct permissions
   → Make sure you own the datasets/dota/ directory

❌ "Directory not found"
   → Create directories first: mkdir -p datasets/dota/raw/
   → Place ZIP files there

❌ "Label and image mismatch"
   → ZIP files may be corrupted
   → Try re-downloading

════════════════════════════════════════════════════════════════════════════════

DOTA-v1 STATISTICS
──────────────────

Training images:   2,806 images (1024×1024 pixels)
Validation images: 1,411 images
Test images:       1,828 images
Total objects:     ~188,282 annotations
Classes:           15 (plane, ship, vehicle, building, etc.)
Format:            Oriented bounding boxes (8 coordinates)

════════════════════════════════════════════════════════════════════════════════

ALTERNATIVE: USE DOTA-v2 INSTEAD
─────────────────────────────────

If you prefer DOTA-v2 (larger, more recent):
  • Same download process from https://captain-whu.github.io/DOTA/
  • Size: ~5GB total (larger images: 4096×4096)
  • More objects: ~1.7M total
  • Run with: python experiments/08_dota_yolo_full_3_seed.py --dota-version v2

════════════════════════════════════════════════════════════════════════════════

USEFUL COMMANDS
───────────────

# Show this guide again
python scripts/dota_v1_help.py

# Get help with setup
python scripts/setup_dota.py --help

# Show setup instructions
python scripts/setup_dota.py --info --version v1

# Verify dataset
python scripts/setup_dota.py --verify --version v1

# Run experiment
python experiments/08_dota_yolo_full_3_seed.py --dota-version v1

════════════════════════════════════════════════════════════════════════════════

NEXT STEPS
──────────

1. Visit: https://captain-whu.github.io/DOTA/
2. Register and download DOTA_v1_split_*.zip files
3. Place in: datasets/dota/raw/
4. Run: python scripts/setup_dota.py --setup --version v1
5. Run: python experiments/08_dota_yolo_full_3_seed.py --dota-version v1
6. Check results: results/08_dota_yolo_full_3_seed/summary_3_seed.json

════════════════════════════════════════════════════════════════════════════════
    """.format(PROJECT_ROOT, PROJECT_ROOT))


def check_existing_files():
    """Check if DOTA files already exist."""
    raw_dir = PROJECT_ROOT / "datasets" / "dota" / "raw"

    print("\n📋 Current Status:")
    print(f"   Raw data directory: {raw_dir}")

    if raw_dir.exists():
        files = list(raw_dir.glob("*.zip"))
        if files:
            print(f"   Found {len(files)} ZIP file(s):")
            for f in files:
                size_gb = f.stat().st_size / (1024**3)
                if size_gb < 0.01:
                    print(f"     ❌ {f.name} ({size_gb:.2f}GB - CORRUPTED, re-download needed)")
                else:
                    print(f"     ✓ {f.name} ({size_gb:.1f}GB)")
        else:
            print("   ❌ No ZIP files found. Download DOTA-v1 first.")
    else:
        print(f"   ❌ Directory doesn't exist. Create with: mkdir -p {raw_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DOTA-v1 Manual Download and Setup Guide"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check download status"
    )

    args = parser.parse_args()

    if args.status:
        check_existing_files()
    else:
        print_manual_download_guide()


if __name__ == "__main__":
    main()
