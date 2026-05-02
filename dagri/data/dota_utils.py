"""
Utilities for downloading and converting DOTA/DOTA-v2 dataset to YOLO format.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np


def download_dota_dataset(output_dir: str, version: str = "v2") -> str:
	"""
	Download DOTA or DOTA-v2 dataset.
	Note: You need to download it manually from https://captain-whu.github.io/DOTA/
	This function validates the expected directory structure.

	Args:
		output_dir: Directory to save the dataset
		version: "v1" or "v2" for DOTA versions

	Returns:
		Path to the extracted dataset directory
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	dota_dir = output_dir / f"DOTA_{version}"

	print(f"DOTA dataset setup for {version}")
	print(f"Expected directory: {dota_dir}")
	print(f"\nTo download DOTA dataset:")
	print(f"1. Visit https://captain-whu.github.io/DOTA/")
	print(f"2. Download DOTA-{version} dataset")
	print(f"3. Extract to {output_dir}")
	print(f"4. Expected structure:")
	print(f"   DOTA_{version}/")
	print(f"   ├── train/")
	print(f"   │   ├── images/")
	print(f"   │   └── labelTxt/")
	print(f"   ├── val/")
	print(f"   │   ├── images/")
	print(f"   │   └── labelTxt/")
	print(f"   └── test/")
	print(f"       └── images/")

	return str(dota_dir)


def dota_to_yolo_bbox(bbox_points: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
	"""
	Convert DOTA's 8-coordinate format to YOLO's center + size format.

	Args:
		bbox_points: 8 coordinates (x1,y1,x2,y2,x3,y3,x4,y4) in image pixels
		img_width: Image width in pixels
		img_height: Image height in pixels

	Returns:
		(center_x_norm, center_y_norm, width_norm, height_norm) normalized to [0,1]
	"""
	# Convert to numpy array and reshape to 4 points
	points = np.array(bbox_points).reshape(4, 2)

	# Calculate bounding box (axis-aligned)
	x_coords = points[:, 0]
	y_coords = points[:, 1]

	x_min = np.min(x_coords)
	x_max = np.max(x_coords)
	y_min = np.min(y_coords)
	y_max = np.max(y_coords)

	# Convert to YOLO format (center coordinates, width, height)
	center_x = (x_min + x_max) / 2.0
	center_y = (y_min + y_max) / 2.0
	width = x_max - x_min
	height = y_max - y_min

	# Normalize to [0, 1]
	center_x_norm = center_x / img_width
	center_y_norm = center_y / img_height
	width_norm = width / img_width
	height_norm = height / img_height

	# Clamp to [0, 1]
	center_x_norm = np.clip(center_x_norm, 0, 1)
	center_y_norm = np.clip(center_y_norm, 0, 1)
	width_norm = np.clip(width_norm, 0, 1)
	height_norm = np.clip(height_norm, 0, 1)

	return float(center_x_norm), float(center_y_norm), float(width_norm), float(height_norm)


def get_class_id(class_name: str, class_names: Dict[str, int]) -> int:
	"""
	Get the class ID for a given class name. Add new classes if needed.

	Args:
		class_name: Name of the class from DOTA
		class_names: Dictionary mapping class names to IDs

	Returns:
		Class ID
	"""
	class_name = class_name.strip()
	if class_name not in class_names:
		class_names[class_name] = len(class_names)
	return class_names[class_name]


def convert_dota_to_yolo(dota_dir: str, output_dir: str, version: str = "v2") -> None:
	"""
	Convert DOTA dataset from native format to YOLO format.

	Args:
		dota_dir: Path to extracted DOTA dataset
		output_dir: Path to save YOLO-formatted dataset
		version: DOTA version ("v1" or "v2")
	"""
	dota_path = Path(dota_dir)
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	# Create YOLO directory structure
	for split in ["train", "val", "test"]:
		(output_path / split / "images").mkdir(parents=True, exist_ok=True)
		(output_path / split / "labels").mkdir(parents=True, exist_ok=True)

	class_names = {}
	split_counts = {"train": 0, "val": 0, "test": 0}

	# Process each split
	for split in ["train", "val", "test"]:
		split_path = dota_path / split
		if not split_path.exists():
			print(f"Skipping {split} - directory not found at {split_path}")
			continue

		images_dir = split_path / "images"
		labels_dir = split_path / "labelTxt"

		if not images_dir.exists():
			print(f"Skipping {split} - images directory not found")
			continue

		print(f"\nProcessing {split} split...")

		# Get all image files
		image_files = sorted(images_dir.glob("*"))
		image_exts = {".png", ".jpg", ".jpeg", ".tif"}
		image_files = [f for f in image_files if f.suffix.lower() in image_exts]

		for image_file in image_files:
			image_name = image_file.stem
			label_file = labels_dir / f"{image_name}.txt" if labels_dir.exists() else None

			# Copy image to YOLO format directory
			output_image = output_path / split / "images" / image_file.name
			shutil.copy2(image_file, output_image)

			# Process labels if they exist
			output_label = output_path / split / "labels" / f"{image_name}.txt"

			if label_file and label_file.exists():
				# Read DOTA format labels
				try:
					from PIL import Image
					img = Image.open(image_file)
					img_width, img_height = img.size
				except Exception as e:
					print(f"Warning: Could not read image dimensions for {image_file}: {e}")
					continue

				yolo_labels = []
				with open(label_file, 'r') as f:
					for line in f:
						line = line.strip()
						if not line or line.startswith("imagesource"):
							continue

						parts = line.split()
						if len(parts) < 9:
							continue

						try:
							# DOTA format: 8 coordinates + difficulty (optional)
							bbox_points = [float(x) for x in parts[:8]]
							class_name = parts[8] if len(parts) > 8 else "object"

							class_id = get_class_id(class_name, class_names)
							cx, cy, w, h = dota_to_yolo_bbox(bbox_points, img_width, img_height)

							yolo_labels.append(f"{class_id} {cx} {cy} {w} {h}")
						except Exception as e:
							print(f"Warning: Could not parse label in {label_file}: {line}")
							continue

				# Write YOLO format labels
				with open(output_label, 'w') as f:
					for label in yolo_labels:
						f.write(label + "\n")
			else:
				# Create empty label file
				output_label.touch()

			split_counts[split] += 1

	# Save class names mapping
	classes_file = output_path / "classes.json"
	class_list = sorted(class_names.items(), key=lambda x: x[1])
	class_names_list = [name for name, _ in class_list]

	with open(classes_file, 'w') as f:
		json.dump({
			"class_names": class_names_list,
			"class_to_id": class_names
		}, f, indent=2)

	print(f"\n✓ Conversion complete!")
	print(f"Train: {split_counts['train']} images")
	print(f"Val: {split_counts['val']} images")
	print(f"Test: {split_counts['test']} images")
	print(f"Classes found: {list(class_names.keys())}")
	print(f"Output saved to: {output_path}")

	return class_names_list


def get_dota_class_names(version: str = "v2") -> List[str]:
	"""
	Get standard DOTA class names.
	"""
	# DOTA and DOTA-v2 have the same 15 classes
	dota_classes = [
		"plane", "baseball-diamond", "bridge", "ground-track-field",
		"small-vehicle", "large-vehicle", "ship", "tennis-court",
		"basketball-court", "storage-tank", "soccer-ball-field",
		"roundabout", "harbor", "swimming-pool", "helicopter"
	]
	return dota_classes
