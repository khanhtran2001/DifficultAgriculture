from __future__ import annotations

import shutil
import sys
import random
from pathlib import Path
from typing import Any

import cv2

from dagri.augmentation.domain_cluster import DomainClusterer
from dagri.augmentation.object_miner import ObjectMiner
from dagri.augmentation.synthesizer import ImageSynthesizer
from dagri.interfaces import AugmentorInterface, DatasetProperties, ScoringResults


class CopyPasteAugmentor(AugmentorInterface):
    def __init__(self, config: dict[str, Any] | None):
        self.config = dict(config or {})

    @staticmethod
    def _parse_optional_positive_int(value: Any) -> int | None:
        if value in (None, "null"):
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None

    @staticmethod
    def _object_reuse_key(image_name: str, object_index: int) -> str:
        return f"{image_name}:{int(object_index)}"

    def _select_background_with_reuse_cap(
        self,
        miner: ObjectMiner,
        background_reuse_counts: dict[str, int],
        max_background_reuse: int | None,
        excluded_names: set[str] | None = None,
    ):
        excluded_names = excluded_names or set()
        eligible_backgrounds = [
            bg
            for bg in miner.background_pool
            if bg.image_name not in excluded_names
            and (
                max_background_reuse is None
                or background_reuse_counts.get(bg.image_name, 0) < max_background_reuse
            )
        ]
        if not eligible_backgrounds:
            return None

        if miner.scoring_mode == "random":
            return miner.rng.choice(eligible_backgrounds)

        weights = miner._build_weights(
            [bg.simg_score for bg in eligible_backgrounds],
            miner.background_weight_mode,
        )
        return miner.rng.choices(eligible_backgrounds, weights=weights, k=1)[0]

    def _apply_object_reuse_cap(
        self,
        compatible_objects,
        object_reuse_counts: dict[str, int],
        max_object_reuse: int | None,
    ):
        if max_object_reuse is None:
            return compatible_objects
        return [
            obj
            for obj in compatible_objects
            if object_reuse_counts.get(self._object_reuse_key(obj.source_image_name, obj.object_index), 0)
            < max_object_reuse
        ]

    def create_new_dataset(
        self,
        initial_dataset_properties: DatasetProperties,
        scoring_results: ScoringResults,
        new_dataset_path: str,
    ) -> DatasetProperties:
        train_images_dir = initial_dataset_properties.train_images_dir
        train_labels_dir = initial_dataset_properties.train_labels_dir
        if not train_images_dir or not train_labels_dir:
            raise ValueError("initial_dataset_properties must include train_images_dir and train_labels_dir")

        output_root = Path(new_dataset_path).resolve()
        train_img_out = output_root / "train" / "images"
        train_lbl_out = output_root / "train" / "labels"
        train_img_out.mkdir(parents=True, exist_ok=True)
        train_lbl_out.mkdir(parents=True, exist_ok=True)

        removed_images, removed_labels = self._remove_previous_augmented_outputs(train_img_out, train_lbl_out)
        if removed_images or removed_labels:
            print(
                "[Augmentor] Cleaned previous synthetic outputs: "
                f"images={removed_images}, labels={removed_labels}"
            )

        self._copy_original_train_split(Path(train_images_dir), Path(train_labels_dir), train_img_out, train_lbl_out)

        mode = str(self.config.get("mode", "difficulty_based_copy_paste")).lower()
        same_image_only = mode in {"same_image_score_based_copy_paste", "same_image_random_copy_paste"}
        scoring_mode = "random" if mode in {"random_copy_paste", "same_image_random_copy_paste"} else "score_targeted"

        dataset_ratio = float(self.config.get("dataset_ratio", self.config.get("relative_multiplier", 0.3)))
        target_density = int(self.config.get("target_density", 12))
        relative_multiplier = float(self.config.get("paste_relative_multiplier", 1.0))
        max_paste_per_image = int(self.config.get("max_paste_objects_per_image", 8))
        use_mask = bool(self.config.get("use_mask", False))
        masks_dir = self.config.get("segmentation_masks_dir")
        if "blending_method" not in self.config:
            raise ValueError(
                "augmentation_config.blending_method is required. "
                "Use 'seamless_clone', 'alpha', 'none', or 'lab_gaussian'."
            )
        blending_method = str(self.config["blending_method"]).lower().strip()
        lab_gaussian_kernel_size = int(self.config.get("lab_gaussian_kernel_size", 15))

        image_extensions = self.config.get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
        auto_k = bool(self.config.get("auto_k", True))
        max_k = int(self.config.get("max_k", 8))

        top_object_fraction = float(self.config.get("top_object_fraction", 0.3))
        object_noise_cap = float(self.config.get("object_noise_cap", 100.0))
        weight_scale = float(self.config.get("weight_scale", 3.0))
        background_weight_mode = str(self.config.get("background_weight_mode", "linear")).lower()
        object_weight_mode = str(self.config.get("object_weight_mode", "linear")).lower()
        max_object_area_px = float(self.config.get("max_object_area_px", 1024.0))
        selection_seed_raw = self.config.get("selection_seed", None)
        selection_rng = random.Random(int(selection_seed_raw)) if selection_seed_raw is not None else random.Random()
        same_image_scale_min = float(self.config.get("same_image_scale_min", 0.75))
        same_image_scale_max = float(self.config.get("same_image_scale_max", 1.25))
        same_image_rotation_deg = float(self.config.get("same_image_rotation_deg", 15.0))
        same_image_min_transformed_area_ratio = float(self.config.get("same_image_min_transformed_area_ratio", 0.5))
        same_image_max_transformed_area_ratio = float(self.config.get("same_image_max_transformed_area_ratio", 2.0))
        same_image_min_transformed_side_px = int(self.config.get("same_image_min_transformed_side_px", 8))
        same_image_max_transformed_side_px_raw = self.config.get("same_image_max_transformed_side_px", None)
        same_image_max_transformed_side_px = (
            None
            if same_image_max_transformed_side_px_raw in (None, "null")
            else int(same_image_max_transformed_side_px_raw)
        )
        max_background_reuse = self._parse_optional_positive_int(self.config.get("max_background_reuse", None))
        max_object_reuse = self._parse_optional_positive_int(self.config.get("max_object_reuse", None))
        if same_image_scale_min > same_image_scale_max:
            same_image_scale_min, same_image_scale_max = same_image_scale_max, same_image_scale_min
        same_image_scale_min = max(0.05, same_image_scale_min)
        same_image_scale_max = max(same_image_scale_min, same_image_scale_max)
        same_image_rotation_deg = max(0.0, same_image_rotation_deg)
        same_image_min_transformed_area_ratio = max(0.01, same_image_min_transformed_area_ratio)
        same_image_max_transformed_area_ratio = max(
            same_image_min_transformed_area_ratio,
            same_image_max_transformed_area_ratio,
        )
        same_image_min_transformed_side_px = max(1, same_image_min_transformed_side_px)
        if same_image_max_transformed_side_px is not None:
            same_image_max_transformed_side_px = max(
                same_image_min_transformed_side_px,
                same_image_max_transformed_side_px,
            )

        clusterer = DomainClusterer(train_images_dir, image_extensions=image_extensions)
        domain_map = clusterer.extract_visual_domains(auto_k=auto_k, max_k=max_k)

        miner = ObjectMiner(
            images_dir=train_images_dir,
            labels_dir=train_labels_dir,
            scoring_results=scoring_results,
            scoring_mode=scoring_mode,
            top_object_fraction=top_object_fraction,
            object_noise_cap=object_noise_cap,
            background_weight_mode=background_weight_mode,
            object_weight_mode=object_weight_mode,
            weight_scale=weight_scale,
            max_object_area_px=max_object_area_px,
            image_extensions=image_extensions,
            rng=selection_rng,
        )
        miner.load_data(domain_map=domain_map)
        if miner.total_images == 0:
            raise RuntimeError("No train images found for augmentation")

        synthesizer = ImageSynthesizer(
            target_density=target_density,
            relative_multiplier=relative_multiplier,
            max_paste_per_image=max_paste_per_image,
            use_mask=use_mask,
            segmentation_masks_dir=masks_dir,
            blending_method=blending_method,
            lab_gaussian_kernel_size=lab_gaussian_kernel_size,
            rng=selection_rng,
        )

        print(f"[Augmentor] {synthesizer.get_mask_config_summary()}")
        print(f"[Augmentor] Blending method: {blending_method}")
        if selection_seed_raw is not None:
            print(f"[Augmentor] Selection seed: {int(selection_seed_raw)}")
        if blending_method == "lab_gaussian":
            print(f"[Augmentor] LAB Gaussian effective kernel size: {synthesizer.lab_gaussian_kernel_size}")

        num_to_generate = max(1, int(miner.total_images * dataset_ratio))
        background_reuse_counts: dict[str, int] = {}
        object_reuse_counts: dict[str, int] = {}
        generated_count = 0
        print(f"[Augmentor] Generating {num_to_generate} new images...")
        for i in range(num_to_generate):
            bg = None
            objects_to_copy = []
            max_background_pick_attempts = max(10, min(200, len(miner.background_pool) * 2))
            attempted_backgrounds: set[str] = set()

            for _ in range(max_background_pick_attempts):
                bg_candidate = self._select_background_with_reuse_cap(
                    miner,
                    background_reuse_counts,
                    max_background_reuse,
                    excluded_names=attempted_backgrounds,
                )
                if bg_candidate is None:
                    break

                attempted_backgrounds.add(bg_candidate.image_name)
                paste_count = synthesizer.calculate_paste_count(len(bg_candidate.existing_boxes))
                if paste_count <= 0:
                    continue

                compatible = miner.get_compatible_objects_for_background(
                    bg_candidate,
                    same_image_only=same_image_only,
                )
                compatible = self._apply_object_reuse_cap(
                    compatible,
                    object_reuse_counts,
                    max_object_reuse,
                )
                candidate_objects = miner.select_objects_to_copy(compatible, paste_count)
                if not candidate_objects:
                    continue

                bg = bg_candidate
                objects_to_copy = candidate_objects
                break

            if bg is None:
                print("\n[Augmentor] Stopped early: no eligible background/object candidates left under reuse caps.")
                break

            scale_factor = 1.0
            rotation_deg = 0.0
            if same_image_only and objects_to_copy:
                scale_factor = selection_rng.uniform(same_image_scale_min, same_image_scale_max)
                rotation_deg = selection_rng.uniform(-same_image_rotation_deg, same_image_rotation_deg)

            aug_image, new_boxes = synthesizer.execute_paste(
                bg,
                objects_to_copy,
                scale_factor=scale_factor,
                rotation_deg=rotation_deg,
                min_transformed_area_ratio=same_image_min_transformed_area_ratio,
                max_transformed_area_ratio=same_image_max_transformed_area_ratio,
                min_transformed_side_px=same_image_min_transformed_side_px,
                max_transformed_side_px=same_image_max_transformed_side_px,
            )

            out_stem = f"aug_{i + 1:04d}_{bg.image_path.stem}"
            out_img_path = train_img_out / f"{out_stem}.jpg"
            out_lbl_path = train_lbl_out / f"{out_stem}.txt"

            cv2.imwrite(str(out_img_path), aug_image)
            merged_boxes = list(bg.existing_boxes) + list(new_boxes)
            self._write_yolo_labels(out_lbl_path, merged_boxes)
            background_reuse_counts[bg.image_name] = background_reuse_counts.get(bg.image_name, 0) + 1
            for obj in objects_to_copy:
                key = self._object_reuse_key(obj.source_image_name, obj.object_index)
                object_reuse_counts[key] = object_reuse_counts.get(key, 0) + 1

            generated_count += 1
            self._print_progress(generated_count, num_to_generate)

        print()
        print(f"[Augmentor] Augmentation generation completed ({generated_count}/{num_to_generate})")

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

    @staticmethod
    def _copy_original_train_split(
        src_images: Path,
        src_labels: Path,
        dst_images: Path,
        dst_labels: Path,
    ) -> None:
        for p in src_images.glob("*"):
            if p.is_file():
                shutil.copy2(p, dst_images / p.name)
        for p in src_labels.glob("*.txt"):
            shutil.copy2(p, dst_labels / p.name)

    @staticmethod
    def _write_yolo_labels(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for cls_id, xc, yc, w, h in boxes:
                f.write(f"{int(cls_id)} {float(xc):.6f} {float(yc):.6f} {float(w):.6f} {float(h):.6f}\n")

    @staticmethod
    def _remove_previous_augmented_outputs(images_dir: Path, labels_dir: Path) -> tuple[int, int]:
        removed_images = 0
        removed_labels = 0

        for p in images_dir.glob("aug_*.*"):
            if p.is_file():
                p.unlink()
                removed_images += 1

        for p in labels_dir.glob("aug_*.txt"):
            if p.is_file():
                p.unlink()
                removed_labels += 1

        return removed_images, removed_labels

    @staticmethod
    def _print_progress(current: int, total: int) -> None:
        if total <= 0:
            return
        width = 30
        ratio = max(0.0, min(float(current) / float(total), 1.0))
        done = int(width * ratio)
        bar = "#" * done + "-" * (width - done)
        sys.stdout.write(f"\r[Augmentor] Progress: [{bar}] {current}/{total} ({ratio * 100:.1f}%)")
        sys.stdout.flush()