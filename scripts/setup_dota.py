#!/usr/bin/env python3
"""
Quick setup script for DOTA experiment.
Helps download and prepare the DOTA dataset.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from dagri.data.dota_utils import (
	download_dota_dataset,
	convert_dota_to_yolo,
	get_dota_class_names
)


def setup_directories(version: str = "v2") -> tuple[str, str]:
	"""Create necessary directories for DOTA experiment."""
	datasets_dir = PROJECT_ROOT / "datasets" / "dota"
	raw_dir = datasets_dir / "raw"
	yolo_dir = datasets_dir / "yolo_format"

	raw_dir.mkdir(parents=True, exist_ok=True)
	yolo_dir.mkdir(parents=True, exist_ok=True)

	print(f"✓ Created directories:")
	print(f"  Raw data:    {raw_dir}")
	print(f"  YOLO format: {yolo_dir}")

	return str(raw_dir), str(yolo_dir)


def check_raw_dataset(raw_dir: str, version: str = "v2") -> bool:
	"""Check if DOTA raw dataset exists."""
	dota_dir = Path(raw_dir) / f"DOTA_{version}"
	train_path = dota_dir / "train" / "images"
	return train_path.exists()


def print_setup_instructions(raw_dir: str, version: str = "v2"):
	"""Print instructions for downloading DOTA."""
	dota_dir = Path(raw_dir) / f"DOTA_{version}"
	print(f"\n{'='*70}")
	print(f"DOTA-{version} Setup Instructions")
	print(f"{'='*70}")
	print(f"\nThe DOTA dataset needs to be downloaded manually.")
	print(f"\n1. Visit: https://captain-whu.github.io/DOTA/")
	print(f"2. Download DOTA-{version} dataset (requires registration)")
	print(f"3. Extract to: {raw_dir}")
	print(f"\nExpected directory structure:")
	print(f"  {dota_dir}/")
	print(f"  ├── train/")
	print(f"  │   ├── images/          (*.jpg, *.png, etc)")
	print(f"  │   └── labelTxt/        (*.txt annotation files)")
	print(f"  ├── val/")
	print(f"  │   ├── images/")
	print(f"  │   └── labelTxt/")
	print(f"  └── test/")
	print(f"      └── images/")
	print(f"\nDOTA-{version} Statistics:")
	if version == "v2":
		print(f"  • Training images: 1,830")
		print(f"  • Validation images: 958")
		print(f"  • Test images: 2,000")
		print(f"  • Total objects: ~1.7M")
		print(f"  • Image size: 4096x4096 (large)")
	else:
		print(f"  • Training images: 2,806")
		print(f"  • Validation images: 1,411")
		print(f"  • Test images: 1,828")
		print(f"  • Total objects: ~188K")
	print(f"\nClasses ({len(get_dota_class_names())}):")
	for i, cls in enumerate(get_dota_class_names(), 1):
		print(f"  {i:2d}. {cls}")
	print(f"{'='*70}\n")


def convert_dataset(raw_dir: str, yolo_dir: str, version: str = "v2"):
	"""Convert DOTA dataset to YOLO format."""
	dota_dir = Path(raw_dir) / f"DOTA_{version}"

	if not dota_dir.exists():
		print(f"❌ DOTA dataset not found at {dota_dir}")
		return False

	print(f"\nConverting DOTA-{version} to YOLO format...")
	print(f"This may take several minutes...\n")

	try:
		convert_dota_to_yolo(str(dota_dir), str(yolo_dir), version=version)
		print(f"\n✓ Conversion complete!")
		return True
	except Exception as e:
		print(f"❌ Conversion failed: {e}")
		return False


def verify_setup(yolo_dir: str) -> bool:
	"""Verify that YOLO formatted dataset is valid."""
	yolo_path = Path(yolo_dir) / "dota_yolo"

	required_dirs = [
		yolo_path / "train" / "images",
		yolo_path / "train" / "labels",
		yolo_path / "val" / "images",
		yolo_path / "val" / "labels",
		yolo_path / "test" / "images",
		yolo_path / "test" / "labels",
	]

	missing = [d for d in required_dirs if not d.exists()]
	if missing:
		print(f"❌ Missing directories:")
		for d in missing:
			print(f"   {d}")
		return False

	# Count files
	train_imgs = len(list((yolo_path / "train" / "images").glob("*")))
	val_imgs = len(list((yolo_path / "val" / "images").glob("*")))
	test_imgs = len(list((yolo_path / "test" / "images").glob("*")))

	print(f"\n✓ Dataset structure verified!")
	print(f"  Train: {train_imgs} images")
	print(f"  Val:   {val_imgs} images")
	print(f"  Test:  {test_imgs} images")

	return True


def main():
	parser = argparse.ArgumentParser(
		description="Setup script for DOTA experiment",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Show setup instructions
  python scripts/setup_dota.py --info

  # Convert existing DOTA dataset to YOLO format
  python scripts/setup_dota.py --convert

  # Full setup with conversion (requires DOTA already downloaded)
  python scripts/setup_dota.py --setup
		"""
	)
	parser.add_argument(
		"--version",
		type=str,
		choices=["v1", "v2"],
		default="v2",
		help="DOTA version to setup (default: v2)"
	)
	parser.add_argument(
		"--info",
		action="store_true",
		help="Show download instructions"
	)
	parser.add_argument(
		"--convert",
		action="store_true",
		help="Convert existing DOTA dataset to YOLO format"
	)
	parser.add_argument(
		"--verify",
		action="store_true",
		help="Verify YOLO dataset is properly formatted"
	)
	parser.add_argument(
		"--setup",
		action="store_true",
		help="Full setup: create dirs, convert, verify"
	)

	args = parser.parse_args()

	# Default action: show info
	if not any([args.info, args.convert, args.verify, args.setup]):
		args.info = True

	raw_dir, yolo_dir = setup_directories(args.version)

	if args.info:
		print_setup_instructions(raw_dir, args.version)

	if args.setup or args.convert:
		if not check_raw_dataset(raw_dir, args.version):
			print(f"❌ DOTA-{args.version} dataset not found at {raw_dir}/DOTA_{args.version}")
			print_setup_instructions(raw_dir, args.version)
			sys.exit(1)

		if convert_dataset(raw_dir, yolo_dir, args.version):
			if args.setup or args.verify:
				verify_setup(yolo_dir)
		else:
			sys.exit(1)

	if args.verify:
		if verify_setup(yolo_dir):
			print(f"\n✓ Ready to run experiment!")
			print(f"  cd {PROJECT_ROOT}")
			print(f"  python experiments/08_dota_yolo_full_3_seed.py --dota-version {args.version}")
		else:
			sys.exit(1)


if __name__ == "__main__":
	main()
