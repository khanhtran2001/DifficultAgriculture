#!/usr/bin/env python3
"""
Convert Global Wheat Head Detection (GWHD) dataset to YOLO format.

Reads CSVs with COCO format bounding boxes and creates YOLO format dataset.
"""
import os
import csv
from pathlib import Path
from PIL import Image
import shutil

def convert_coco_to_yolo(coco_box, img_width, img_height):
    """
    Convert COCO format (x1, y1, x2, y2) to YOLO format (x_center, y_center, width, height).
    All values normalized to 0-1.
    """
    x1, y1, x2, y2 = coco_box
    
    # Calculate center and dimensions
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    
    # Normalize to 0-1
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def process_gwhd_dataset(
    raw_dir,
    output_dir,
    train_csv,
    val_csv,
    test_csv,
):
    """Process GWHD dataset and convert to YOLO format."""
    
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Class mapping - GWHD only has wheat heads (class 0)
    class_id = 0  # Only one class: wheat head
    
    # Process each split
    splits = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv,
    }
    
    for split, csv_file in splits.items():
        csv_path = raw_dir / csv_file
        images_dir = raw_dir / "images"
        
        print(f"\nProcessing {split} split from {csv_file}...")
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                image_name = row['image_name']
                boxes_string = row['BoxesString']
                domain = row['domain']
                
                # Find and copy image
                img_path = images_dir / image_name
                if not img_path.exists():
                    print(f"  WARNING: Image not found: {image_name}")
                    continue
                
                # Get image dimensions
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    print(f"  WARNING: Could not read image {image_name}: {e}")
                    continue
                
                # Copy image to output directory
                output_img_path = output_dir / split / "images" / image_name
                shutil.copy2(img_path, output_img_path)
                
                # Parse and convert bounding boxes
                label_path = output_dir / split / "labels" / image_name.replace('.png', '.txt')
                with open(label_path, 'w') as label_file:
                    if boxes_string and boxes_string.strip():
                        boxes = boxes_string.split(';')
                        for box in boxes:
                            try:
                                coords = [int(x) for x in box.split()]
                                if len(coords) == 4:
                                    x_center, y_center, width, height = convert_coco_to_yolo(
                                        coords, img_width, img_height
                                    )
                                    # YOLO format: class_id x_center y_center width height
                                    label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            except ValueError as e:
                                print(f"  WARNING: Could not parse box in {image_name}: {box}")
                
                count += 1
                if count % 500 == 0:
                    print(f"  Processed {count} images...")
        
        print(f"  Completed {split}: {count} images")
    
    # Create data.yaml
    data_yaml = f"""path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['wheat_head']
"""
    
    with open(output_dir / "data.yaml", 'w') as f:
        f.write(data_yaml)
    
    print(f"\nDataset conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"data.yaml created at: {output_dir / 'data.yaml'}")

if __name__ == "__main__":
    raw_dir = "/home/khanh/Projects/DifficultyAgri/datasets/global_wheat_head/raw_v2/gwhd_2021"
    output_dir = "/home/khanh/Projects/DifficultyAgri/datasets/global_wheat_head/yolo_format_v2"
    
    process_gwhd_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        train_csv="competition_train.csv",
        val_csv="competition_val.csv",
        test_csv="competition_test.csv",
    )
