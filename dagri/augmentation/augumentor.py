from __future__ import annotations

import shutil
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

        self._copy_original_train_split(Path(train_images_dir), Path(train_labels_dir), train_img_out, train_lbl_out)

        mode = str(self.config.get("mode", "difficulty_based_copy_paste")).lower()
        scoring_mode = "random" if mode == "random_copy_paste" else "score_targeted"

        dataset_ratio = float(self.config.get("dataset_ratio", self.config.get("relative_multiplier", 0.3)))
        target_density = int(self.config.get("target_density", 12))
        relative_multiplier = float(self.config.get("paste_relative_multiplier", 1.0))
        max_paste_per_image = int(self.config.get("max_paste_objects_per_image", 8))
        use_mask = bool(self.config.get("use_mask", False))
        masks_dir = self.config.get("segmentation_masks_dir")

        image_extensions = self.config.get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
        auto_k = bool(self.config.get("auto_k", True))
        max_k = int(self.config.get("max_k", 8))

        top_object_fraction = float(self.config.get("top_object_fraction", 0.3))
        object_noise_cap = float(self.config.get("object_noise_cap", 100.0))
        weight_scale = float(self.config.get("weight_scale", 3.0))
        background_weight_mode = str(self.config.get("background_weight_mode", "linear")).lower()
        object_weight_mode = str(self.config.get("object_weight_mode", "linear")).lower()
        max_object_area_px = float(self.config.get("max_object_area_px", 1024.0))

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
        )

        num_to_generate = max(1, int(miner.total_images * dataset_ratio))
        for i in range(num_to_generate):
            bg = miner.select_background_image()
            paste_count = synthesizer.calculate_paste_count(len(bg.existing_boxes))
            compatible = miner.get_compatible_objects(bg)
            objects_to_copy = miner.select_objects_to_copy(compatible, paste_count)
            aug_image, new_boxes = synthesizer.execute_paste(bg, objects_to_copy)

            out_stem = f"aug_{i + 1:04d}_{bg.image_path.stem}"
            out_img_path = train_img_out / f"{out_stem}.jpg"
            out_lbl_path = train_lbl_out / f"{out_stem}.txt"

            cv2.imwrite(str(out_img_path), aug_image)
            merged_boxes = list(bg.existing_boxes) + list(new_boxes)
            self._write_yolo_labels(out_lbl_path, merged_boxes)

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