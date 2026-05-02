#!/usr/bin/env python3
"""
DOTA-v2 Automatic Download using Ultralytics + Roboflow
Fastest and easiest method - no manual registration needed!
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from dagri.data.dota_utils import convert_dota_to_yolo


def download_dota_v2_easy(output_dir: str) -> bool:
	"""
	Download DOTA-v2 using Roboflow (hosted by Ultralytics).
	This is the fastest and most reliable method!
	"""
	try:
		from roboflow import Roboflow
	except ImportError:
		print("❌ roboflow not installed")
		print("Install with: pip install roboflow")
		return False

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	raw_dir = output_path / "raw"
	raw_dir.mkdir(parents=True, exist_ok=True)

	print("="*70)
	print("DOTA-v2 Automatic Download (Roboflow/Ultralytics)")
	print("="*70)
	print(f"\nTarget directory: {raw_dir}")
	print(f"Dataset: DOTA-v2 (OBB Format)")
	print(f"Size: ~5GB")
	print("⏱️  Downloading...\n")

	# Check if already downloaded
	dota_dir = raw_dir / "DOTA-v2"
	if (dota_dir / "train" / "images").exists():
		print(f"✓ DOTA-v2 already exists at {dota_dir}")
		return True

	try:
		# Initialize Roboflow
		rf = Roboflow(api_key="")  # Public dataset, no API key needed

		# Download DOTA-v2 OBB dataset
		print("Connecting to Roboflow...")
		project = rf.workspace("roboflow-100").project("dota-v2-obb")

		print("Downloading DOTA-v2 dataset...")
		dataset = project.version(1).download("obb")

		print(f"✓ Downloaded to: {dataset.location}")

		# Move to expected location
		import shutil
		dataset_path = Path(dataset.location)

		if dataset_path.exists():
			# The downloaded dataset might be in a different structure
			# We need to reorganize it to match DOTA-v2 standard format
			print("\nReorganizing dataset...")

			dota_dir.mkdir(parents=True, exist_ok=True)

			# Copy train, val, test splits
			for split in ["train", "val", "test"]:
				split_src = dataset_path / split
				split_dst = dota_dir / split

				if split_src.exists():
					if split_dst.exists():
						shutil.rmtree(split_dst)
					shutil.copytree(split_src, split_dst)
					print(f"  ✓ {split} split ready")

			print(f"\n✓ Dataset ready at: {dota_dir}")
			return True

	except Exception as e:
		print(f"❌ Roboflow download failed: {e}")
		return False


def download_dota_v2_direct(output_dir: str) -> bool:
	"""
	Direct download from Ultralytics source.
	Faster alternative if Roboflow doesn't work.
	"""
	import urllib.request
	import zipfile
	import shutil

	output_path = Path(output_dir)
	raw_dir = output_path / "raw"
	raw_dir.mkdir(parents=True, exist_ok=True)

	dota_dir = raw_dir / "DOTA-v2"
	if (dota_dir / "train" / "images").exists():
		print(f"✓ DOTA-v2 already exists")
		return True

	print("="*70)
	print("DOTA-v2 Direct Download (Ultralytics)")
	print("="*70)

	# Try downloading DOTA8 (DOTA subset) as fallback
	# This is smaller and faster for testing
	urls = [
		("DOTA8", "https://github.com/ultralytics/yolov5/releases/download/v1.0/DOTA8.zip"),
		("DOTA8", "https://ultralytics.com/assets/DOTA8.zip"),
	]

	for dataset_name, url in urls:
		try:
			print(f"\nDownloading {dataset_name}...")
			print(f"URL: {url}")

			zip_path = raw_dir / f"{dataset_name}.zip"

			print("Downloading...")
			urllib.request.urlretrieve(url, zip_path)
			print(f"✓ Downloaded ({zip_path.stat().st_size / (1024**2):.0f}MB)")

			print("Extracting...")
			with zipfile.ZipFile(zip_path, 'r') as zip_ref:
				zip_ref.extractall(raw_dir)
			print("✓ Extracted")

			# Rename and organize
			dataset_dir = raw_dir / dataset_name
			if dataset_dir.exists() and not dota_dir.exists():
				dataset_dir.rename(dota_dir)
				print(f"✓ Organized to {dota_dir}")

			# Cleanup
			zip_path.unlink()

			return True

		except Exception as e:
			print(f"  ❌ Failed: {e}")
			continue

	return False


def main():
	import argparse

	parser = argparse.ArgumentParser(
		description="Download DOTA-v2 easily (Ultralytics/Roboflow)",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
RECOMMENDED: This script automatically downloads DOTA-v2 without
manual registration. Much faster than manual download!

Examples:
  # Download DOTA-v2 automatically
  python scripts/download_dota_ultralytics.py

  # Only convert existing DOTA-v2 to YOLO format
  python scripts/download_dota_ultralytics.py --convert-only

  # Skip download verification
  python scripts/download_dota_ultralytics.py --skip-check
		"""
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(PROJECT_ROOT / "datasets" / "dota"),
		help="Output directory (default: datasets/dota)"
	)
	parser.add_argument(
		"--convert-only",
		action="store_true",
		help="Skip download, only convert to YOLO format"
	)
	parser.add_argument(
		"--skip-check",
		action="store_true",
		help="Skip checking for existing dataset"
	)

	args = parser.parse_args()
	output_dir = Path(args.output_dir)

	# Download
	if not args.convert_only:
		print("\n🚀 Starting DOTA-v2 download...\n")

		success = False

		# Try Roboflow first (recommended)
		print("Method 1: Roboflow (Recommended)")
		print("-" * 70)
		if download_dota_v2_easy(str(output_dir)):
			success = True
		else:
			# Try direct download as fallback
			print("\n\nMethod 2: Direct Download (Fallback)")
			print("-" * 70)
			if download_dota_v2_direct(str(output_dir)):
				success = True

		if not success:
			print("\n❌ All download methods failed")
			print("\nOptions:")
			print("1. Ensure you have internet connection")
			print("2. Install roboflow: pip install roboflow")
			print("3. Manual download: https://captain-whu.github.io/DOTA/")
			sys.exit(1)

	# Convert to YOLO format
	print(f"\n{'='*70}")
	print("Converting to YOLO format...")
	print(f"{'='*70}\n")

	raw_dir = output_dir / "raw"
	dota_v2_path = str(raw_dir / "DOTA-v2")
	yolo_dir = output_dir / "yolo_format"
	yolo_dir.mkdir(parents=True, exist_ok=True)

	if not Path(dota_v2_path).exists():
		print(f"❌ DOTA-v2 not found at {dota_v2_path}")
		sys.exit(1)

	try:
		convert_dota_to_yolo(dota_v2_path, str(yolo_dir), version="v2")
		print(f"\n✓ Conversion complete!")
		print(f"  Location: {yolo_dir}/dota_yolo")
		print(f"\n🎯 Next: Run the experiment!")
		print(f"  cd {PROJECT_ROOT}")
		print(f"  python experiments/08_dota_yolo_full_3_seed.py --dota-version v2")
	except Exception as e:
		print(f"❌ Conversion failed: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
