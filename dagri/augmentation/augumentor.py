from __future__ import annotations

import json
import shutil
import sys
import random
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import cv2
import numpy as np

from dagri.augmentation.synthesizer import ImageSynthesizer
from dagri.interfaces import AugmentorInterface, DatasetProperties, ScoringResults


def boxes_overlap(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> bool:
    """Check if two bounding boxes overlap. Boxes are in (x1, y1, x2, y2) pixel format."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)


@dataclass
class ImageData:
    """Simple image metadata"""
    name: str
    path: Path
    boxes: list[tuple[int, float, float, float, float]]
    score: float = 0.0


@dataclass
class ObjectData:
    """Simple object metadata"""
    image_name: str
    image_path: Path
    object_index: int
    bbox: tuple[int, float, float, float, float]
    score: float = 0.0




class CopyPasteAugmentor(AugmentorInterface):
    """Simple copy-paste augmentor with random/score-guided selection and reuse caps"""

    def __init__(self, config: dict[str, Any] | None):
        self.config = dict(config or {})

    def create_new_dataset(
        self,
        initial_dataset_properties: DatasetProperties,
        scoring_results: ScoringResults,
        new_dataset_path: str,
    ) -> DatasetProperties:
        """Create augmented dataset with copy-paste augmentation"""
        train_images_dir = initial_dataset_properties.train_images_dir
        train_labels_dir = initial_dataset_properties.train_labels_dir
        if not train_images_dir or not train_labels_dir:
            raise ValueError("initial_dataset_properties must include train_images_dir and train_labels_dir")

        # Setup output directories
        output_root = Path(new_dataset_path).resolve()
        train_img_out = output_root / "train" / "images"
        train_lbl_out = output_root / "train" / "labels"
        train_meta_out = output_root / "train" / "metadata"
        train_img_out.mkdir(parents=True, exist_ok=True)
        train_lbl_out.mkdir(parents=True, exist_ok=True)
        train_meta_out.mkdir(parents=True, exist_ok=True)

        # Clean previous augmented outputs
        removed_images, removed_labels, removed_metadata = self._remove_previous_augmented_outputs(
            train_img_out,
            train_lbl_out,
            train_meta_out,
        )
        if removed_images or removed_labels or removed_metadata:
            print(
                f"[Augmentor] Cleaned previous outputs: images={removed_images}, labels={removed_labels}, metadata={removed_metadata}"
            )

        # Copy original training split
        self._copy_original_train_split(
            Path(train_images_dir), Path(train_labels_dir), train_img_out, train_lbl_out
        )

        # Parse config
        use_score = self.config.get("use_score_guidance")
        reverse_score_guidance = bool(self.config.get("reverse_score_guidance"))
        dataset_ratio = float(self.config.get("dataset_ratio"))
        min_objects_per_image = int(self.config.get("min_objects_per_image", 1))
        max_objects_per_image = int(self.config.get("max_objects_per_image", self.config.get("num_objects_per_image", 3)))
        score_weight_function = str(self.config.get("score_weight_function", "linear")).strip().lower()
        score_alpha = float(self.config.get("score_alpha", 1.0))
        same_image_only = bool(self.config.get("same_image_only"))
        max_image_reuse = self._normalize_reuse_cap(self.config.get("max_image_reuse"))
        max_object_reuse = self._normalize_reuse_cap(self.config.get("max_object_reuse"))
        scale_min = float(self.config.get("scale_min"))
        scale_max = float(self.config.get("scale_max"))
        rotation_deg_max = float(self.config.get("rotation_deg_max"))
        min_object_area_px = float(self.config.get("min_object_area_px"))
        blending_method = str(self.config.get("blending_method", "none")).strip().lower()
        lab_gaussian_kernel_size = int(self.config.get("lab_gaussian_kernel_size", 5))
        avoid_overlap = bool(self.config.get("avoid_overlap"))
        placement_control = bool(self.config.get("placement_control"))
        placement_margin_px = int(self.config.get("placement_margin_px"))
        random_placement_attempts = int(self.config.get("random_placement_attempts"))
        use_jiggle_placement = bool(self.config.get("use_jiggle_placement"))
        image_extensions = self.config.get("image_extensions", [".jpg", ".jpeg", ".png"])
        selection_seed = self.config.get("selection_seed")

        # New: Boundary-based selection method
        augmentation_method = str(self.config.get("augmentation_method")).strip().lower()
        selection_group = str(self.config.get("group")).strip().lower()

        rng = random.Random(int(selection_seed)) if selection_seed not in (None, "null") else random.Random()

        # Load images and objects
        images = self._load_images(
            Path(train_images_dir),
            Path(train_labels_dir),
            image_extensions,
            scoring_results if use_score else None,
        )
        objects = self._load_objects(
            images,
            scoring_results if use_score else None,
            min_object_area_px=min_object_area_px,
        )

        if not images or not objects:
            raise RuntimeError(f"No images ({len(images)}) or objects ({len(objects)}) found for augmentation")

        print(f"[Augmentor] Loaded {len(images)} images and {len(objects)} objects")
        if use_score:
            print(f"[Augmentor] Using score-guided selection (method={augmentation_method})")
        print(
            f"[Augmentor] Config: dataset_ratio={dataset_ratio}, "
            f"min_objects={min_objects_per_image}, max_objects={max_objects_per_image}"
        )
        if use_score:
            if augmentation_method == "boundary":
                print(f"[Augmentor] Boundary selection: group={selection_group}")
            else:
                reverse_str = " (REVERSED)" if reverse_score_guidance else ""
                print(
                    f"[Augmentor] Score weighting: function={score_weight_function}, alpha={score_alpha}{reverse_str}"
                )
        if min_object_area_px > 0:
            print(f"[Augmentor] Min object area filter: {min_object_area_px:.1f} px^2")
        print(f"[Augmentor] Object source mode: {'same-image-only' if same_image_only else 'whole-pool'}")
        if max_image_reuse:
            print(f"[Augmentor] Max image reuse: {max_image_reuse}")
        if max_object_reuse:
            print(f"[Augmentor] Max object reuse: {max_object_reuse}")
        print(f"[Augmentor] Blending method: {blending_method}")

        synthesizer = ImageSynthesizer(
            use_mask=False,
            segmentation_masks_dir=None,
            blending_method=blending_method,
            lab_gaussian_kernel_size=lab_gaussian_kernel_size,
            rng=rng,
        )

        # Generate augmented images
        num_to_generate = max(1, int(len(images) * dataset_ratio))
        image_reuse_counts: dict[str, int] = {}
        object_reuse_counts: dict[str, int] = {}
        generated_count = 0

        print(f"[Augmentor] Generating {num_to_generate} augmented images...")

        for i in range(num_to_generate):
            # Select background image
            bg_image = self._select_image(
                images,
                image_reuse_counts,
                max_image_reuse,
                use_score,
                score_weight_function,
                score_alpha,
                reverse_score_guidance,
                rng,
            )
            if bg_image is None:
                print("\n[Augmentor] Stopped: no eligible background images left under reuse caps")
                break

            # Select objects to copy
            low = max(1, min(min_objects_per_image, max_objects_per_image))
            high = max(1, max(min_objects_per_image, max_objects_per_image))
            num_objects = rng.randint(low, high)

            object_pool = objects
            if same_image_only:
                object_pool = [obj for obj in objects if obj.image_name == bg_image.name]

            selected_objects = self._select_objects(
                object_pool,
                object_reuse_counts,
                max_object_reuse,
                num_objects,
                use_score,
                score_weight_function,
                score_alpha,
                reverse_score_guidance,
                rng,
            )
            if not selected_objects:
                continue

            # Apply random transform
            scale_factor = rng.uniform(scale_min, scale_max)
            rotation_deg = rng.uniform(-rotation_deg_max, rotation_deg_max)

            # Paste objects onto background
            aug_image, new_boxes = self._paste_objects(
                bg_image,
                selected_objects,
                scale_factor,
                rotation_deg,
                synthesizer,
                avoid_overlap,
                placement_margin_px,
                random_placement_attempts,
                use_jiggle_placement,
                placement_control,
                rng,
            )

            # Save augmented image
            out_stem = f"aug_{i + 1:04d}_{bg_image.path.stem}"
            out_img_path = train_img_out / f"{out_stem}.jpg"
            out_lbl_path = train_lbl_out / f"{out_stem}.txt"

            cv2.imwrite(str(out_img_path), aug_image)
            merged_boxes = list(bg_image.boxes) + new_boxes
            self._write_yolo_labels(out_lbl_path, merged_boxes)
            self._write_augmented_metadata(
                train_meta_out / f"{out_stem}.json",
                bg_image,
                selected_objects,
                new_boxes,
                use_score,
                score_weight_function,
                score_alpha,
                reverse_score_guidance,
                same_image_only,
                blending_method,
            )

            # Update reuse counts
            bg_key = str(bg_image.path)
            image_reuse_counts[bg_key] = image_reuse_counts.get(bg_key, 0) + 1
            for obj in selected_objects:
                key = f"{obj.image_name}:{obj.object_index}"
                object_reuse_counts[key] = object_reuse_counts.get(key, 0) + 1

            generated_count += 1
            self._print_progress(generated_count, num_to_generate)

        print()
        print(f"[Augmentor] Completed: generated {generated_count}/{num_to_generate} augmented images")

        return DatasetProperties(
            root_dir=str(output_root),
            num_classes=initial_dataset_properties.num_classes,
            class_names=initial_dataset_properties.class_names,
            train_mask_dir=initial_dataset_properties.train_mask_dir,
            train_images_dir=str(train_img_out),
            train_labels_dir=str(train_lbl_out),
            val_images_dir=initial_dataset_properties.val_images_dir,
            val_labels_dir=initial_dataset_properties.val_labels_dir,
            test_images_dir=initial_dataset_properties.test_images_dir,
            test_labels_dir=initial_dataset_properties.test_labels_dir,
        )

    def _load_images(
        self,
        images_dir: Path,
        labels_dir: Path,
        image_extensions: list[str],
        scoring_results: ScoringResults | None = None,
    ) -> list[ImageData]:
        """Load images and their labels"""
        images = []
        score_map = self._build_score_map(scoring_results) if scoring_results else {}

        image_paths = sorted(
            [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]
        )

        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt"
            boxes = self._read_yolo_labels(label_path)
            score = score_map.get(image_path.name, 0.0)

            images.append(
                ImageData(
                    name=image_path.name,
                    path=image_path,
                    boxes=boxes,
                    score=score,
                )
            )

        return images

    def _load_objects(
        self,
        images: list[ImageData],
        scoring_results: ScoringResults | None = None,
        min_object_area_px: float = 0.0,
    ) -> list[ObjectData]:
        """Extract all objects from images"""
        objects = []
        score_map = self._build_object_score_map(scoring_results) if scoring_results else {}
        min_area = max(float(min_object_area_px), 0.0)
        filtered_small_objects = 0

        for image in images:
            image_array = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
            if image_array is None:
                continue
            image_h, image_w = image_array.shape[:2]

            for obj_idx, bbox in enumerate(image.boxes):
                _, _, _, bbox_w, bbox_h = bbox
                bbox_area_px = max(float(bbox_w), 0.0) * max(float(bbox_h), 0.0) * image_w * image_h
                if bbox_area_px < min_area:
                    filtered_small_objects += 1
                    continue

                score_key = f"{image.name}:{obj_idx}"
                score = score_map.get(score_key, 0.0)

                objects.append(
                    ObjectData(
                        image_name=image.name,
                        image_path=image.path,
                        object_index=obj_idx,
                        bbox=bbox,
                        score=score,
                    )
                )

        if min_area > 0 and filtered_small_objects > 0:
            print(
                f"[Augmentor] Filtered out {filtered_small_objects} objects below area threshold ({min_area:.1f} px^2)"
            )

        return objects

    def _select_image(
        self,
        images: list[ImageData],
        reuse_counts: dict[str, int],
        max_reuse: int | None,
        use_score: bool,
        score_weight_function: str,
        score_alpha: float,
        reverse_score_guidance: bool,
        rng: random.Random,
    ) -> ImageData | None:
        """Select a background image, respecting reuse caps"""
        available = [
            img for img in images
            if max_reuse is None or reuse_counts.get(str(img.path), 0) < max_reuse
        ]

        if not available:
            return None

        if not use_score:
            return rng.choice(available)

        # Score-guided selection
        weights = self._scores_to_weights(
            [img.score for img in available],
            score_weight_function,
            score_alpha,
            reverse_score_guidance,
        )
        return rng.choices(available, weights=weights, k=1)[0]

    @staticmethod
    def _normalize_reuse_cap(value: Any) -> int | None:
        """Normalize reuse cap from config. Returns None when disabled/invalid."""
        if value is None:
            return None
        try:
            cap = int(value)
        except (TypeError, ValueError):
            return None
        return cap if cap > 0 else None

    def _select_objects(
        self,
        objects: list[ObjectData],
        reuse_counts: dict[str, int],
        max_reuse: int | None,
        num_to_select: int,
        use_score: bool,
        score_weight_function: str,
        score_alpha: float,
        reverse_score_guidance: bool,
        rng: random.Random,
        augmentation_method: str = "score",
        selection_group: str = "medium",
    ) -> list[ObjectData]:
        """Select objects to copy, allowing within-image repeats while enforcing reuse caps."""
        if not objects or num_to_select <= 0:
            return []

        selected: list[ObjectData] = []
        selected_counts: dict[str, int] = {}

        for _ in range(num_to_select):
            available = []
            for obj in objects:
                key = f"{obj.image_name}:{obj.object_index}"
                current_count = reuse_counts.get(key, 0) + selected_counts.get(key, 0)
                if max_reuse is None or current_count < max_reuse:
                    available.append(obj)

            if not available:
                break

            # Apply boundary filtering if using boundary method
            if use_score and augmentation_method == "boundary":
                available = self._filter_by_boundary(available, selection_group)
                if not available:
                    break
                picked = rng.choice(available)
            elif use_score:
                weights = self._scores_to_weights(
                    [obj.score for obj in available],
                    score_weight_function,
                    score_alpha,
                    reverse_score_guidance,
                )
                picked = rng.choices(available, weights=weights, k=1)[0]
            else:
                picked = rng.choice(available)

            selected.append(picked)
            key = f"{picked.image_name}:{picked.object_index}"
            selected_counts[key] = selected_counts.get(key, 0) + 1

        return selected

    @staticmethod
    def _filter_by_boundary(objects: list[ObjectData], selection_group: str) -> list[ObjectData]:
        """Filter objects into 3 groups (low, medium, high) based on difficulty scores.
        
        Divides objects into 3 equal-size groups when sorted by score:
        - 'low': lowest third of objects (easier)
        - 'medium': middle third of objects
        - 'high': highest third of objects (harder)
        
        Args:
            objects: List of ObjectData to filter
            selection_group: One of 'low', 'medium', 'high'
            
        Returns:
            List of objects in the selected group
        """
        if not objects:
            return []
        
        group = selection_group.strip().lower()
        if group not in {"low", "medium", "high"}:
            group = "medium"
        
        # Sort by score in ascending order (low to high difficulty)
        sorted_objects = sorted(objects, key=lambda obj: obj.score)
        
        # Divide into 3 groups of equal size
        group_size = len(sorted_objects) // 3
        remainder = len(sorted_objects) % 3
        
        if group == "low":
            # Lowest third (easiest objects)
            return sorted_objects[:group_size]
        elif group == "high":
            # Highest third (hardest objects)
            return sorted_objects[-(group_size + remainder):]
        else:  # medium
            # Middle third
            start = group_size
            end = 2 * group_size + remainder
            return sorted_objects[start:end]

    @staticmethod
    def _scores_to_weights(scores: list[float], function_name: str, alpha: float, reverse: bool = False) -> list[float]:
        """Convert raw scores to positive sampling weights.

        Supported functions:
        - linear: weight = normalized_score^alpha
        - exponential: weight = exp(alpha * normalized_score) - 1
        
        If reverse=True, uses 1 - normalized_score instead (lower scores get higher weights).
        """
        if not scores:
            return []

        cleaned = [max(float(s), 0.0) for s in scores]
        min_s = min(cleaned)
        max_s = max(cleaned)
        if max_s > min_s:
            normalized = [(s - min_s) / (max_s - min_s) for s in cleaned]
        else:
            normalized = [1.0 for _ in cleaned]

        # Reverse scores if requested (so lower scores get selected more often)
        if reverse:
            normalized = [1.0 - x for x in normalized]

        fn = function_name.strip().lower()
        if fn == "liner":
            fn = "linear"
        if fn not in {"linear", "exponential"}:
            fn = "linear"

        a = max(1.0, float(alpha))
        eps = 1e-6
        if fn == "exponential":
            return [float(np.exp(a * x) - 1.0 + eps) for x in normalized]

        return [float((x ** a) + eps) for x in normalized]

    def _paste_objects(
        self,
        bg_image: ImageData,
        objects: list[ObjectData],
        scale_factor: float,
        rotation_deg: float,
        synthesizer: ImageSynthesizer,
        avoid_overlap: bool,
        placement_margin_px: int,
        random_placement_attempts: int,
        use_jiggle_placement: bool,
        placement_control: bool,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
        """Paste objects onto background image with resize, rotation, and collision avoidance"""
        bg_img = cv2.imread(str(bg_image.path), cv2.IMREAD_COLOR)
        if bg_img is None:
            raise RuntimeError(f"Failed to read background image: {bg_image.path}")

        bg_h, bg_w = bg_img.shape[:2]
        new_boxes: list[tuple[int, float, float, float, float]] = []

        # Convert existing boxes to pixel coordinates for collision detection
        existing_boxes_px: list[tuple[float, float, float, float]] = []
        for cls_id, x_center, y_center, width, height in bg_image.boxes:
            x1 = (x_center - width / 2) * bg_w
            y1 = (y_center - height / 2) * bg_h
            x2 = (x_center + width / 2) * bg_w
            y2 = (y_center + height / 2) * bg_h
            existing_boxes_px.append((x1, y1, x2, y2))

        for obj in objects:
            # Extract object from source image
            src_img = cv2.imread(str(obj.image_path), cv2.IMREAD_COLOR)
            if src_img is None:
                continue

            src_h, src_w = src_img.shape[:2]
            cls_id, x_center, y_center, width, height = obj.bbox

            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - width / 2) * src_w)
            y1 = int((y_center - height / 2) * src_h)
            x2 = int((x_center + width / 2) * src_w)
            y2 = int((y_center + height / 2) * src_h)

            # Clip to valid range
            x1 = max(0, min(x1, src_w - 1))
            y1 = max(0, min(y1, src_h - 1))
            x2 = max(x1 + 1, min(x2, src_w))
            y2 = max(y1 + 1, min(y2, src_h))

            obj_crop = src_img[y1:y2, x1:x2]
            if obj_crop.size == 0:
                continue

            # Reuse blending/transform implementation from ImageSynthesizer.
            obj_h, obj_w = obj_crop.shape[:2]
            obj_mask = np.full((obj_h, obj_w), 255, dtype=np.uint8)
            obj_pixels, obj_mask = synthesizer.transform_object_patch(
                obj_crop,
                obj_mask,
                scale_factor=scale_factor,
                rotation_deg=rotation_deg,
            )

            obj_h_pasted, obj_w_pasted = obj_pixels.shape[:2]
            if obj_h_pasted <= 0 or obj_w_pasted <= 0:
                continue

            # Find valid placement that avoids collision with existing boxes
            paste_x, paste_y = self._find_valid_placement(
                bg_w,
                bg_h,
                obj_w_pasted,
                obj_h_pasted,
                existing_boxes_px,
                avoid_overlap,
                placement_margin_px,
                random_placement_attempts,
                use_jiggle_placement,
                placement_control,
                rng,
            )

            if paste_x is None or paste_y is None:
                # Could not find valid placement
                continue

            # Paste object onto background
            if paste_y + obj_h_pasted <= bg_h and paste_x + obj_w_pasted <= bg_w:
                debug_tag = f"bg={bg_image.name}|src={obj.image_name}|obj={obj.object_index}"
                bg_img, bbox_xyxy = synthesizer.blend_and_paste(
                    bg_img,
                    obj_pixels,
                    obj_mask,
                    (paste_x, paste_y),
                    debug_tag=debug_tag,
                )
                x1_px, y1_px, x2_px, y2_px = bbox_xyxy
                if x2_px <= x1_px or y2_px <= y1_px:
                    continue

                # Convert back to YOLO format
                new_x_center = ((x1_px + x2_px) / 2) / bg_w
                new_y_center = ((y1_px + y2_px) / 2) / bg_h
                new_width = max(0.0, (x2_px - x1_px) / bg_w)
                new_height = max(0.0, (y2_px - y1_px) / bg_h)

                new_boxes.append((cls_id, new_x_center, new_y_center, new_width, new_height))

                # Add new box to existing boxes for future collision checks
                existing_boxes_px.append((float(paste_x), float(paste_y), 
                                         float(paste_x + obj_w_pasted), float(paste_y + obj_h_pasted)))

        return bg_img, new_boxes

    def _find_valid_placement(
        self,
        bg_w: int,
        bg_h: int,
        obj_w: int,
        obj_h: int,
        existing_boxes_px: list[tuple[float, float, float, float]],
        avoid_overlap: bool,
        margin: int,
        random_attempts: int,
        use_jiggle_placement: bool,
        placement_control: bool,
        rng: random.Random,
    ) -> tuple[int | None, int | None]:
        """Find a valid placement position that avoids collision with existing boxes.
        
        Strategy:
        1. If placement control is enabled, try positions around existing boxes first
        2. Otherwise try random placements first
        3. Fall back to the other strategy if needed
        4. Return None if no valid position found
        """
        max_x = max(0, bg_w - obj_w)
        max_y = max(0, bg_h - obj_h)

        def try_random() -> tuple[int | None, int | None]:
            for _ in range(max(1, random_attempts)):
                paste_x = rng.randint(0, max_x) if max_x > 0 else 0
                paste_y = rng.randint(0, max_y) if max_y > 0 else 0

                if not avoid_overlap:
                    return paste_x, paste_y

                new_box = (float(paste_x), float(paste_y), float(paste_x + obj_w), float(paste_y + obj_h))
                if not self._collision_with_existing(new_box, existing_boxes_px, margin):
                    return paste_x, paste_y
            return None, None

        def try_jiggle() -> tuple[int | None, int | None]:
            if not (avoid_overlap and use_jiggle_placement and existing_boxes_px):
                return None, None
            candidates = []
            for ex_x1, ex_y1, ex_x2, ex_y2 in existing_boxes_px:
                # Try positions: above, below, left, right of existing box
                jiggle_positions = [
                    # Above
                    ((ex_x1 + ex_x2) / 2 - obj_w / 2, ex_y1 - obj_h - margin),
                    # Below
                    ((ex_x1 + ex_x2) / 2 - obj_w / 2, ex_y2 + margin),
                    # Left
                    (ex_x1 - obj_w - margin, (ex_y1 + ex_y2) / 2 - obj_h / 2),
                    # Right
                    (ex_x2 + margin, (ex_y1 + ex_y2) / 2 - obj_h / 2),
                    # Top-left corner
                    (ex_x1 - obj_w - margin, ex_y1 - obj_h - margin),
                    # Top-right corner
                    (ex_x2 + margin, ex_y1 - obj_h - margin),
                    # Bottom-left corner
                    (ex_x1 - obj_w - margin, ex_y2 + margin),
                    # Bottom-right corner
                    (ex_x2 + margin, ex_y2 + margin),
                ]

                for px, py in jiggle_positions:
                    paste_x = max(0, min(int(px), bg_w - obj_w))
                    paste_y = max(0, min(int(py), bg_h - obj_h))

                    if paste_y + obj_h <= bg_h and paste_x + obj_w <= bg_w:
                        new_box = (float(paste_x), float(paste_y), float(paste_x + obj_w), float(paste_y + obj_h))
                        if not self._collision_with_existing(new_box, existing_boxes_px, margin):
                            candidates.append((paste_x, paste_y))

            if candidates:
                paste_x, paste_y = rng.choice(candidates)
                return paste_x, paste_y

            return None, None

        if placement_control:
            paste_x, paste_y = try_jiggle()
            if paste_x is not None and paste_y is not None:
                return paste_x, paste_y
            return try_random()

        paste_x, paste_y = try_random()
        if paste_x is not None and paste_y is not None:
            return paste_x, paste_y

        return try_jiggle()

    @staticmethod
    def _collision_with_existing(
        new_box: tuple[float, float, float, float],
        existing_boxes_px: list[tuple[float, float, float, float]],
        margin: int = 0,
    ) -> bool:
        """Check if new box collides with any existing box"""
        for ex_box in existing_boxes_px:
            # Apply margin by expanding existing box
            ex_x1, ex_y1, ex_x2, ex_y2 = ex_box
            ex_x1_expanded = ex_x1 - margin
            ex_y1_expanded = ex_y1 - margin
            ex_x2_expanded = ex_x2 + margin
            ex_y2_expanded = ex_y2 + margin

            if boxes_overlap(new_box, (ex_x1_expanded, ex_y1_expanded, ex_x2_expanded, ex_y2_expanded)):
                return True

        return False

    @staticmethod
    def _build_score_map(scoring_results: ScoringResults) -> dict[str, float]:
        """Build image name -> score mapping"""
        score_map = {}
        for img in scoring_results.image_difficulties:
            p = Path(img.image_path)
            score_map[p.name] = float(img.difficulty_score)
            score_map[p.stem] = float(img.difficulty_score)
        return score_map

    @staticmethod
    def _build_object_score_map(scoring_results: ScoringResults) -> dict[str, float]:
        """Build (image_name:object_id) -> score mapping"""
        score_map = {}
        for img in scoring_results.image_difficulties:
            p = Path(img.image_path)
            for obj in img.objects_score:
                score_map[f"{p.name}:{int(obj.object_id)}"] = float(obj.difficulty_score)
                score_map[f"{p.stem}:{int(obj.object_id)}"] = float(obj.difficulty_score)
        return score_map

    @staticmethod
    def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
        """Read YOLO format labels"""
        if not label_path.exists():
            return []
        boxes = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                x_center, y_center, width, height = map(float, parts[1:])
                boxes.append((cls_id, x_center, y_center, width, height))
        return boxes

    def _copy_original_train_split(
        self,
        src_images: Path,
        src_labels: Path,
        dst_images: Path,
        dst_labels: Path,
    ) -> None:
        """Copy original training images and labels to output directory"""
        for p in src_images.glob("*"):
            if p.is_file():
                shutil.copy2(p, dst_images / p.name)
        for p in src_labels.glob("*.txt"):
            shutil.copy2(p, dst_labels / p.name)

    @staticmethod
    def _write_yolo_labels(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
        """Write boxes in YOLO format"""
        with open(path, "w", encoding="utf-8") as f:
            for cls_id, xc, yc, w, h in boxes:
                f.write(f"{int(cls_id)} {float(xc):.6f} {float(yc):.6f} {float(w):.6f} {float(h):.6f}\n")

    @staticmethod
    def _write_augmented_metadata(
        path: Path,
        bg_image: ImageData,
        selected_objects: list[ObjectData],
        new_boxes: list[tuple[int, float, float, float, float]],
        use_score: bool,
        score_weight_function: str,
        score_alpha: float,
        reverse_score_guidance: bool,
        same_image_only: bool,
        blending_method: str,
    ) -> None:
        """Write per-augmented-image provenance metadata for notebook analysis."""
        payload = {
            "background_image_name": bg_image.name,
            "background_image_path": str(bg_image.path),
            "background_box_count": len(bg_image.boxes),
            "selected_object_count": len(selected_objects),
            "pasted_object_count": len(new_boxes),
            "use_score_guidance": bool(use_score),
            "reverse_score_guidance": bool(reverse_score_guidance),
            "same_image_only": bool(same_image_only),
            "blending_method": str(blending_method),
            "score_weight_function": str(score_weight_function),
            "score_alpha": float(score_alpha),
            "selected_objects": [
                {
                    "source_image_name": obj.image_name,
                    "source_image_path": str(obj.image_path),
                    "source_object_index": int(obj.object_index),
                    "class_id": int(obj.bbox[0]),
                    "score": float(obj.score),
                    "bbox_yolo": [
                        float(obj.bbox[1]),
                        float(obj.bbox[2]),
                        float(obj.bbox[3]),
                        float(obj.bbox[4]),
                    ],
                }
                for obj in selected_objects
            ],
            "pasted_boxes": [
                {
                    "class_id": int(cls_id),
                    "x_center": float(xc),
                    "y_center": float(yc),
                    "width": float(w),
                    "height": float(h),
                }
                for cls_id, xc, yc, w, h in new_boxes
            ],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _remove_previous_augmented_outputs(images_dir: Path, labels_dir: Path, metadata_dir: Path) -> tuple[int, int, int]:
        """Remove previously generated augmented outputs"""
        removed_images = 0
        removed_labels = 0
        removed_metadata = 0

        for p in images_dir.glob("aug_*.*"):
            if p.is_file():
                p.unlink()
                removed_images += 1

        for p in labels_dir.glob("aug_*.txt"):
            if p.is_file():
                p.unlink()
                removed_labels += 1

        for p in metadata_dir.glob("aug_*.json"):
            if p.is_file():
                p.unlink()
                removed_metadata += 1

        return removed_images, removed_labels, removed_metadata

    @staticmethod
    def _print_progress(current: int, total: int) -> None:
        """Print progress bar"""
        if total <= 0:
            return
        width = 30
        ratio = max(0.0, min(float(current) / float(total), 1.0))
        done = int(width * ratio)
        bar = "#" * done + "-" * (width - done)
        sys.stdout.write(f"\r[Augmentor] Progress: [{bar}] {current}/{total} ({ratio * 100:.1f}%)")
        sys.stdout.flush()