import json
import os
from glob import glob
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def create_coco_gt_json_from_yolo(
    images_dir: str,
    output_json_path: str,
    labels_dir: str | None = None,
    class_names: list[str] | None = None,
) -> str:
    """
    Create a COCO detection ground-truth JSON from YOLO txt labels.

    Args:
        images_dir: Directory containing images.
        output_json_path: Path where COCO json will be written.
        labels_dir: Directory containing YOLO label txt files.
            If None, defaults to sibling folder named "labels".
        class_names: Optional class names. If None, names are generated as class_0, class_1, ...

    Returns:
        Path to the generated COCO JSON file.
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    if labels_dir is None:
        labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        image_paths.extend(glob(os.path.join(images_dir, ext)))
        image_paths.extend(glob(os.path.join(images_dir, ext.upper())))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise ValueError(f"No images found in: {images_dir}")

    images = []
    annotations = []
    categories_map = {}
    next_ann_id = 1

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        file_name = os.path.basename(img_path)
        stem = os.path.splitext(file_name)[0]
        image_id = stem

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(w),
                "height": int(h),
            }
        )

        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])

                x = (x_center - bw / 2.0) * w
                y = (y_center - bh / 2.0) * h
                box_w = bw * w
                box_h = bh * h

                x = max(0.0, min(x, w - 1.0))
                y = max(0.0, min(y, h - 1.0))
                box_w = max(0.0, min(box_w, w - x))
                box_h = max(0.0, min(box_h, h - y))

                coco_cat_id = cls + 1
                categories_map[coco_cat_id] = cls

                annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": image_id,
                        "category_id": coco_cat_id,
                        "bbox": [x, y, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                    }
                )
                next_ann_id += 1

    if class_names is None:
        if categories_map:
            max_cls = max(categories_map.values())
            class_names = [f"class_{i}" for i in range(max_cls + 1)]
        else:
            class_names = []

    categories = []
    for cls_idx, cls_name in enumerate(class_names):
        categories.append({"id": cls_idx + 1, "name": cls_name})

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)

    return output_json_path
