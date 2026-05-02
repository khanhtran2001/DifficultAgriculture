#!/usr/bin/env python3
"""
DOTA-v1 Dataset Download and Format Conversion Script
Downloads DOTA-v1 dataset and converts to YOLO format automatically.
"""

import os
import sys
import zipfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from dagri.data.dota_utils import convert_dota_to_yolo, get_dota_class_names


def download_with_gdown(url: str, output_path: str) -> bool:
	"""Download file using gdown (for Google Drive links)."""
	try:
		import gdown
		print(f"\nDownloading from Google Drive: {url}")
		gdown.download(url, output_path, quiet=False)
		return os.path.exists(output_path)
	except ImportError:
		print("❌ gdown not installed. Install with: pip install gdown")
		return False


def download_with_wget(url: str, output_path: str) -> bool:
	"""Download file using wget."""
	try:
		print(f"\nDownloading: {url}")
		subprocess.run(
			["wget", "-c", url, "-O", output_path],
			check=True,
			timeout=3600  # 1 hour timeout
		)
		return os.path.exists(output_path)
	except (subprocess.CalledProcessError, FileNotFoundError):
		return False


def download_with_curl(url: str, output_path: str) -> bool:
	"""Download file using curl."""
	try:
		print(f"\nDownloading: {url}")
		subprocess.run(
			["curl", "-C", "-", "-L", url, "-o", output_path],
			check=True,
			timeout=3600  # 1 hour timeout
		)
		return os.path.exists(output_path)
	except (subprocess.CalledProcessError, FileNotFoundError):
		return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
	"""Extract ZIP file."""
	try:
		print(f"\nExtracting {zip_path}...")
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(extract_to)
		print(f"✓ Extracted to {extract_to}")
		return True
	except Exception as e:
		print(f"❌ Extraction failed: {e}")
		return False


def download_dota_v1(output_dir: str) -> Optional[str]:
	"""
	Download DOTA-v1 dataset.

	Returns:
		Path to extracted DOTA_v1 directory, or None if failed
	"""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	# DOTA-v1 files
	# Note: These URLs might require authentication or may need to be updated
	dota_v1_parts = {
		"DOTA_v1_split_1of2.zip": "https://captain-whu.github.io/DOTA/download/DOTA_v1_split_1of2.zip",
		"DOTA_v1_split_2of2.zip": "https://captain-whu.github.io/DOTA/download/DOTA_v1_split_2of2.zip",
	}

	print("="*70)
	print("DOTA-v1 Dataset Download")
	print("="*70)
	print(f"\nTarget directory: {output_path}")
	print(f"Total size: ~8GB (2 parts, ~4GB each)")
	print("\nThis will download in parts and extract automatically.\n")

	# Check if already downloaded
	dota_dir = output_path / "DOTA_v1"
	if (dota_dir / "train" / "images").exists():
		print(f"✓ DOTA-v1 already exists at {dota_dir}")
		return str(dota_dir)

	# Download each part
	downloaded_parts = []
	for filename, url in dota_v1_parts.items():
		zip_path = output_path / filename

		# Skip if already downloaded
		if zip_path.exists():
			print(f"✓ {filename} already downloaded")
			downloaded_parts.append(str(zip_path))
			continue

		print(f"\n{'='*70}")
		print(f"Downloading {filename}")
		print(f"{'='*70}")
		print(f"URL: {url}")
		print(f"Size: ~4GB")
		print(f"To: {zip_path}\n")

		# Try different download methods
		success = False
		for downloader, name in [
			(download_with_wget, "wget"),
			(download_with_curl, "curl"),
			(download_with_gdown, "gdown"),
		]:
			print(f"Trying {name}...")
			if downloader(url, str(zip_path)):
				print(f"✓ Downloaded with {name}")
				success = True
				break

		if not success:
			print(f"❌ Failed to download {filename}")
			print("\nManual download required:")
			print(f"1. Visit: {url}")
			print(f"2. Download to: {zip_path}")
			print(f"3. Re-run this script")
			return None

		downloaded_parts.append(str(zip_path))

	# Extract all parts
	print(f"\n{'='*70}")
	print("Extracting files...")
	print(f"{'='*70}\n")

	extract_dir = output_path / "DOTA_v1_extracted"
	extract_dir.mkdir(parents=True, exist_ok=True)

	for zip_path in downloaded_parts:
		if not extract_zip(zip_path, str(extract_dir)):
			return None

	# Reorganize extracted files to standard DOTA-v1 structure
	print("\nOrganizing files to DOTA-v1 standard structure...")

	# DOTA-v1 uses train/val/test split
	if not (dota_dir / "train" / "images").exists():
		dota_dir.mkdir(parents=True, exist_ok=True)

		# Look for extracted directories
		extracted_items = list(extract_dir.glob("*"))

		if len(extracted_items) == 1 and extracted_items[0].is_dir():
			# Single root directory
			src = extracted_items[0]
		else:
			src = extract_dir

		# Copy structure
		for split in ["train", "val", "test"]:
			src_split = src / split
			dst_split = dota_dir / split

			if src_split.exists():
				if dst_split.exists():
					shutil.rmtree(dst_split)
				shutil.copytree(src_split, dst_split)
				print(f"✓ Copied {split} split")

	# Clean up extracted files
	if extract_dir.exists():
		shutil.rmtree(extract_dir)
		print(f"✓ Cleaned up temporary files")

	print(f"\n✓ DOTA-v1 dataset ready at: {dota_dir}")
	return str(dota_dir)


def main():
	parser = argparse.ArgumentParser(
		description="Download and format DOTA-v1 dataset",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Download and convert DOTA-v1
  python download_dota_v1.py

  # Download to custom location
  python download_dota_v1.py --output-dir /custom/path

  # Skip download, only convert existing DOTA-v1
  python download_dota_v1.py --skip-download

  # Download only (no conversion)
  python download_dota_v1.py --download-only
		"""
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(PROJECT_ROOT / "datasets" / "dota"),
		help="Output directory for dataset (default: datasets/dota)"
	)
	parser.add_argument(
		"--skip-download",
		action="store_true",
		help="Skip download, only convert existing DOTA-v1"
	)
	parser.add_argument(
		"--download-only",
		action="store_true",
		help="Download only, don't convert to YOLO format"
	)

	args = parser.parse_args()

	output_dir = Path(args.output_dir)
	raw_dir = output_dir / "raw"

	# Download
	if not args.skip_download:
		dota_v1_path = download_dota_v1(str(raw_dir))
		if not dota_v1_path:
			print("\n❌ Download failed. Please download manually from:")
			print("  https://captain-whu.github.io/DOTA/")
			sys.exit(1)
	else:
		dota_v1_path = str(raw_dir / "DOTA_v1")
		if not Path(dota_v1_path).exists():
			print(f"❌ DOTA-v1 not found at {dota_v1_path}")
			print("Please download it first or omit --skip-download")
			sys.exit(1)

	if args.download_only:
		print(f"\n✓ Download complete. DOTA-v1 at: {dota_v1_path}")
		return

	# Convert to YOLO format
	print(f"\n{'='*70}")
	print("Converting DOTA-v1 to YOLO format...")
	print(f"{'='*70}\n")

	yolo_dir = output_dir / "yolo_format"
	yolo_dir.mkdir(parents=True, exist_ok=True)

	try:
		convert_dota_to_yolo(dota_v1_path, str(yolo_dir), version="v1")
		print(f"\n✓ Conversion complete!")
		print(f"YOLO format dataset at: {yolo_dir}/dota_yolo")
		print(f"\nReady to run experiment:")
		print(f"  python experiments/08_dota_yolo_full_3_seed.py --dota-version v1")
	except Exception as e:
		print(f"❌ Conversion failed: {e}")
		sys.exit(1)


if __name__ == "__main__":
	main()
