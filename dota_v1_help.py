#!/usr/bin/env python3
"""
Simple DOTA-v1 Download Helper
Downloads DOTA-v1 from official sources and formats to YOLO.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def print_instructions():
	"""Print download instructions."""
	print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   DOTA-v1 Dataset Download Instructions                    ║
╚════════════════════════════════════════════════════════════════════════════╝

The DOTA-v1 dataset requires manual download from the official website.

OPTION 1: Download from Official Source (Recommended)
─────────────────────────────────────────────────────
1. Visit: https://captain-whu.github.io/DOTA/
2. Look for "DOTA_v1_split_1of2.zip" and "DOTA_v1_split_2of2.zip"
3. Download both files (each ~4GB)
4. Extract to: {}/datasets/dota/raw/
5. Run: python download_dota_v1.py --skip-download


OPTION 2: Direct Download (if available)
─────────────────────────────────────────
Run this command to attempt automatic download:

  python download_dota_v1.py

This will try to download and convert automatically.


OPTION 3: Use Ultralytics Dataset
──────────────────────────────────
The Ultralytics documentation provides DOTA-v2, not v1.
Visit: https://docs.ultralytics.com/datasets/obb/dota-v2/

For v1, use the official source above.


After download/extraction, format conversion happens automatically:
  • Converts DOTA's 8-point boxes → YOLO format
  • Creates train/val/test splits
  • Generates classes.json mapping
  • Ready for experiment

Questions?
──────────
1. Check: {} exists
2. Try manual download from official website
3. Ensure ZIP files are in: {}/datasets/dota/raw/
4. Run: python setup_dota.py --verify
	""".format(PROJECT_ROOT, PROJECT_ROOT, PROJECT_ROOT))


def main():
	import argparse

	parser = argparse.ArgumentParser(
		description="DOTA-v1 Download Helper"
	)
	parser.add_argument(
		"--instructions",
		action="store_true",
		help="Show download instructions"
	)
	parser.add_argument(
		"--auto",
		action="store_true",
		help="Attempt automatic download"
	)

	args = parser.parse_args()

	if args.instructions or (not args.auto):
		print_instructions()
		return

	if args.auto:
		print("Attempting automatic download...")
		print("Note: You may need to manually download from the official website.")
		print(f"\nTrying to run: python {PROJECT_ROOT}/download_dota_v1.py")
		try:
			subprocess.run(
				[sys.executable, str(PROJECT_ROOT / "download_dota_v1.py")],
				check=True
			)
		except Exception as e:
			print(f"Automatic download failed: {e}")
			print_instructions()


if __name__ == "__main__":
	main()
