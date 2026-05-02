"""
VisDrone Dataset Utilities - Convert to YOLO format
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def get_visdrone_classes() -> List[str]:
	"""
	Get standard VisDrone object classes.
	VisDrone has 10 main classes for detection task.
	"""
	return [
		"pedestrian",      # 0
		"person",          # 1
		"car",             # 2
		"van",             # 3
		"truck",           # 4
		"tricycle",        # 5
		"awning-tricycle", # 6
		"bus",             # 7
		"motor",           # 8
		"bicycle"          # 9
	]


def visdrone_to_yolo_bbox(
	x: float, y: float, w: float, h: float,
	img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
	"""
	Convert VisDrone's (x, y, width, height) format to YOLO's (cx, cy, w, h) normalized format.

	Args:
		x, y: Top-left corner coordinates
		w, h: Width and height
		img_width, img_height: Image dimensions

	Returns:
		(center_x_norm, center_y_norm, width_norm, height_norm) normalized to [0,1]
	"""
	# Convert to center coordinates
	center_x = x + w / 2.0
	center_y = y + h / 2.0

	# Normalize to [0, 1]
	center_x_norm = center_x / img_width
	center_y_norm = center_y / img_height
	width_norm = w / img_width
	height_norm = h / img_height

	# Clamp to [0, 1]
	center_x_norm = np.clip(center_x_norm, 0, 1)
	center_y_norm = np.clip(center_y_norm, 0, 1)
	width_norm = np.clip(width_norm, 0, 1)
	height_norm = np.clip(height_norm, 0, 1)

	return float(center_x_norm), float(center_y_norm), float(width_norm), float(height_norm)


def convert_visdrone_to_yolo(
	raw_dir: str,
	output_dir: str
) -> Dict[str, int]:
	"""
	Convert VisDrone dataset to YOLO format.

	Args:
		raw_dir: Path to raw VisDrone dataset
		output_dir: Path to save YOLO-formatted dataset

	Returns:
		Dictionary with counts per split
	"""
	raw_path = Path(raw_dir)
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	# Map VisDrone splits to our train/val/test
	split_mapping = {
		"VisDrone2019-DET-train": "train",
		"VisDrone2019-DET-val": "val",
		"VisDrone2019-DET-test-dev": "test",
		# Note: test-challenge has no annotations, so we skip it
	}

	counts = {"train": 0, "val": 0, "test": 0}

	# Create YOLO directory structure
	for split in ["train", "val", "test"]:
		(output_path / split / "images").mkdir(parents=True, exist_ok=True)
		(output_path / split / "labels").mkdir(parents=True, exist_ok=True)

	class_names = {}
	class_names_list = get_visdrone_classes()
	for i, name in enumerate(class_names_list):
		class_names[i] = name

	# Process each split
	for visdrone_split, output_split in split_mapping.items():
		split_path = raw_path / visdrone_split

		if not split_path.exists():
			print(f"⚠️  {visdrone_split} not found, skipping...")
			continue

		images_dir = split_path / "images"
		annotations_dir = split_path / "annotations"

		print(f"\nProcessing {visdrone_split}...")

		# Get all image files
		image_files = sorted(images_dir.glob("*"))
		image_exts = {".jpg", ".jpeg", ".png"}
		image_files = [f for f in image_files if f.suffix.lower() in image_exts]

		for image_file in image_files:
			image_name = image_file.stem
			ann_file = annotations_dir / f"{image_name}.txt" if annotations_dir.exists() else None

			# Copy image
			output_image = output_path / output_split / "images" / image_file.name
			import shutil
			shutil.copy2(image_file, output_image)

			# Process annotations
			output_label = output_path / output_split / "labels" / f"{image_name}.txt"

			if ann_file and ann_file.exists():
				# Get image dimensions
				try:
					from PIL import Image
					img = Image.open(image_file)
					img_width, img_height = img.size
				except Exception as e:
					print(f"Warning: Could not read image {image_file}: {e}")
					continue

				# Read VisDrone annotations
				yolo_labels = []
				with open(ann_file, 'r') as f:
					for line in f:
						line = line.strip()
						if not line:
							continue

						parts = line.split(',')
						if len(parts) < 5:
							continue

						try:
							# VisDrone format: x,y,w,h,score,class_id,truncation,occlusion
							x = float(parts[0])
							y = float(parts[1])
							w = float(parts[2])
							h = float(parts[3])

							# Skip invalid boxes
							if w <= 0 or h <= 0:
								continue

							# Class ID (if provided)
							if len(parts) >= 6:
								class_id = int(parts[5])
							else:
								class_id = 0

							# Clamp class ID
							if class_id < 0 or class_id >= len(class_names_list):
								class_id = 0

							# Convert to YOLO format
							cx, cy, w_norm, h_norm = visdrone_to_yolo_bbox(x, y, w, h, img_width, img_height)

							yolo_labels.append(f"{class_id} {cx} {cy} {w_norm} {h_norm}")

						except Exception as e:
							print(f"Warning: Could not parse annotation in {ann_file}: {line}")
							continue

				# Write YOLO labels
				with open(output_label, 'w') as f:
					for label in yolo_labels:
						f.write(label + "\n")
			else:
				# Create empty label file
				output_label.touch()

			counts[output_split] += 1

	# Save class names mapping
	classes_file = output_path / "classes.json"
	with open(classes_file, 'w') as f:
		json.dump({
			"class_names": class_names_list,
			"class_to_id": {name: i for i, name in enumerate(class_names_list)}
		}, f, indent=2)

	print(f"\n✓ Conversion complete!")
	print(f"Train: {counts['train']} images")
	print(f"Val: {counts['val']} images")
	print(f"Test: {counts['test']} images")
	print(f"Classes: {len(class_names_list)}")
	print(f"Output: {output_path}")

	return counts
