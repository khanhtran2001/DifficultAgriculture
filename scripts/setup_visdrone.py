#!/usr/bin/env python3
"""
VisDrone Dataset Setup and Conversion to YOLO Format
Converts raw VisDrone dataset to YOLO format for DifficultyAgri experiments.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from dagri.data.visdrone_utils import convert_visdrone_to_yolo, get_visdrone_classes


def verify_raw_dataset(raw_dir: str) -> bool:
	"""
	Verify that raw VisDrone dataset has correct structure.
	"""
	raw_path = Path(raw_dir)

	print("\n" + "="*70)
	print("Verifying VisDrone Raw Dataset")
	print("="*70)

	required_splits = {
		"VisDrone2019-DET-train": {"images": True, "annotations": True},
		"VisDrone2019-DET-val": {"images": True, "annotations": True},
		"VisDrone2019-DET-test-dev": {"images": True, "annotations": True},
	}

	all_valid = True
	for split, requirements in required_splits.items():
		split_path = raw_path / split
		print(f"\n{split}:")

		if not split_path.exists():
			print(f"  ❌ Directory not found")
			all_valid = False
			continue

		# Check images
		images_dir = split_path / "images"
		if images_dir.exists():
			img_count = len(list(images_dir.glob("*")))
			print(f"  ✓ images: {img_count} files")
		else:
			print(f"  ❌ images directory missing")
			all_valid = False

		# Check annotations
		annotations_dir = split_path / "annotations"
		if requirements["annotations"]:
			if annotations_dir.exists():
				ann_count = len(list(annotations_dir.glob("*.txt")))
				print(f"  ✓ annotations: {ann_count} files")

				# Verify matching
				if img_count != ann_count:
					print(f"  ⚠️  WARNING: Image count ({img_count}) != Annotation count ({ann_count})")
					all_valid = False
			else:
				print(f"  ❌ annotations directory missing")
				all_valid = False

	if all_valid:
		print(f"\n✓ VisDrone dataset structure is valid!")
	else:
		print(f"\n❌ VisDrone dataset has issues")

	return all_valid


def main():
	import argparse

	parser = argparse.ArgumentParser(
		description="Setup and convert VisDrone dataset to YOLO format",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Verify raw dataset structure
  python scripts/setup_visdrone.py --verify

  # Convert raw VisDrone to YOLO format
  python scripts/setup_visdrone.py --convert

  # Full setup (verify + convert)
  python scripts/setup_visdrone.py --setup
		"""
	)
	parser.add_argument(
		"--raw-dir",
		type=str,
		default=str(PROJECT_ROOT / "datasets" / "VisDrone" / "raw"),
		help="Raw VisDrone dataset directory (default: datasets/VisDrone/raw)"
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(PROJECT_ROOT / "datasets" / "VisDrone" / "yolo_format"),
		help="Output directory for YOLO-formatted dataset"
	)
	parser.add_argument(
		"--verify",
		action="store_true",
		help="Verify raw dataset structure"
	)
	parser.add_argument(
		"--convert",
		action="store_true",
		help="Convert raw dataset to YOLO format"
	)
	parser.add_argument(
		"--setup",
		action="store_true",
		help="Full setup: verify and convert"
	)

	args = parser.parse_args()

	# Default: show info
	if not any([args.verify, args.convert, args.setup]):
		print("\n" + "="*70)
		print("VisDrone Dataset Setup")
		print("="*70)
		print("\nVisDrone is a large-scale benchmark for object detection")
		print("in drone-captured images with 10 object classes.")
		print("\nDataset Info:")
		print("  Training:   6,471 images")
		print("  Validation: 548 images")
		print("  Test:       1,610 images (test-dev with annotations)")
		print(f"\nClasses ({len(get_visdrone_classes())}):")
		for i, cls in enumerate(get_visdrone_classes(), 1):
			print(f"  {i:2d}. {cls}")
		print("\nUsage:")
		print("  python scripts/setup_visdrone.py --verify      # Check dataset structure")
		print("  python scripts/setup_visdrone.py --convert     # Convert to YOLO format")
		print("  python scripts/setup_visdrone.py --setup       # Full setup")
		return

	raw_dir = Path(args.raw_dir)
	output_dir = Path(args.output_dir)

	# Verify
	if args.verify or args.setup:
		if not verify_raw_dataset(str(raw_dir)):
			sys.exit(1)

	# Convert
	if args.convert or args.setup:
		if not raw_dir.exists():
			print(f"\n❌ Raw directory not found: {raw_dir}")
			print("\nExpected structure:")
			print(f"  {raw_dir}/")
			print(f"  ├── VisDrone2019-DET-train/")
			print(f"  ├── VisDrone2019-DET-val/")
			print(f"  └── VisDrone2019-DET-test-dev/")
			sys.exit(1)

		print("\n" + "="*70)
		print("Converting to YOLO Format")
		print("="*70)

		output_dir.mkdir(parents=True, exist_ok=True)
		yolo_dir = output_dir / "visdrone_yolo"

		try:
			convert_visdrone_to_yolo(str(raw_dir), str(yolo_dir))
			print(f"\n✓ Conversion complete!")
			print(f"  Output: {yolo_dir}")
			print(f"\n🎯 Ready to run experiment!")
			print(f"  cd {PROJECT_ROOT}")
			print(f"  python experiments/09_visdrone_yolo_full_3_seed.py")
		except Exception as e:
			print(f"\n❌ Conversion failed: {e}")
			import traceback
			traceback.print_exc()
			sys.exit(1)


if __name__ == "__main__":
	main()
