from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from dagri.augmentation.object_miner import BackgroundImageData, MinedObject


class ImageSynthesizer:
	def __init__(
		self,
		target_density: int = 12,
		relative_multiplier: float = 1.0,
		max_paste_per_image: int = 8,
		use_mask: bool = False,
		segmentation_masks_dir: str | None = None,
	):
		self.target_density = int(target_density)
		self.relative_multiplier = float(relative_multiplier)
		self.max_paste_per_image = int(max_paste_per_image)
		self.use_mask = bool(use_mask)
		self.segmentation_masks_dir = Path(segmentation_masks_dir).resolve() if segmentation_masks_dir else None
		self._source_mask_cache: dict[str, np.ndarray | None] = {}

	def _load_source_mask(self, source_image_path: Path) -> np.ndarray | None:
		if not self.use_mask or self.segmentation_masks_dir is None:
			return None

		cache_key = str(source_image_path)
		if cache_key in self._source_mask_cache:
			return self._source_mask_cache[cache_key]

		stem = source_image_path.stem
		candidates = [
			self.segmentation_masks_dir / f"{stem}.png",
			self.segmentation_masks_dir / stem / "all_apples_mask.png",
		]

		selected = None
		for c in candidates:
			if c.exists():
				selected = c
				break

		if selected is None:
			self._source_mask_cache[cache_key] = None
			return None

		m = cv2.imread(str(selected), cv2.IMREAD_UNCHANGED)
		if m is None:
			self._source_mask_cache[cache_key] = None
			return None
		if m.ndim == 3:
			m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
		self._source_mask_cache[cache_key] = m
		return m

	@staticmethod
	def _build_object_mask_from_crop(mask_crop: np.ndarray | None, obj_h: int, obj_w: int) -> np.ndarray:
		full_mask = np.full((obj_h, obj_w), 255, dtype=np.uint8)
		if mask_crop is None or mask_crop.shape[:2] != (obj_h, obj_w):
			return full_mask

		positive_pixels = mask_crop[mask_crop > 0]
		if positive_pixels.size == 0:
			return full_mask

		uniq, counts = np.unique(positive_pixels, return_counts=True)
		if uniq.size == 1:
			return (mask_crop > 0).astype(np.uint8) * 255

		dominant = uniq[int(np.argmax(counts))]
		return (mask_crop == dominant).astype(np.uint8) * 255

	def calculate_paste_count(self, current_apples: int) -> int:
		remaining = max(self.target_density - current_apples, 0)
		relative_limit = int(max(current_apples * self.relative_multiplier, 1))
		return min(remaining, relative_limit, self.max_paste_per_image)

	def find_placement_coordinates(
		self,
		bg_existing_bboxes: list[tuple[int, float, float, float, float]],
		image_h: int,
		image_w: int,
		obj_h: int,
		obj_w: int,
		max_overlap: float = 0.3,
		max_attempts: int = 15,
	) -> tuple[int, int]:
		max_x = max(image_w - obj_w, 0)
		max_y = max(image_h - obj_h, 0)

		if not bg_existing_bboxes:
			return random.randint(0, max_x), random.randint(0, max_y)

		pixel_boxes = []
		for bbox in bg_existing_bboxes:
			_, xc, yc, w, h = bbox
			px_w, px_h = w * image_w, h * image_h
			x1 = (xc * image_w) - (px_w / 2)
			y1 = (yc * image_h) - (px_h / 2)
			pixel_boxes.append([x1, y1, x1 + px_w, y1 + px_h])

		for _ in range(max_attempts):
			tx1, ty1, tx2, ty2 = random.choice(pixel_boxes)
			tw = tx2 - tx1
			th = ty2 - ty1
			jitter_x = int(max(tw * 1.5, obj_w * 2))
			jitter_y = int(max(th * 1.5, obj_h * 2))

			prop_x = int(random.uniform(tx1 - jitter_x, tx2 + jitter_x))
			prop_y = int(random.uniform(ty1 - jitter_y, ty2 + jitter_y))
			prop_x = min(max(prop_x, 0), max_x)
			prop_y = min(max(prop_y, 0), max_y)

			prop_x2 = prop_x + obj_w
			prop_y2 = prop_y + obj_h
			prop_area = max(1.0, obj_w * obj_h)

			safe = True
			for ex1, ey1, ex2, ey2 in pixel_boxes:
				ix1 = max(prop_x, ex1)
				iy1 = max(prop_y, ey1)
				ix2 = min(prop_x2, ex2)
				iy2 = min(prop_y2, ey2)

				if ix1 < ix2 and iy1 < iy2:
					inter = (ix2 - ix1) * (iy2 - iy1)
					ex_area = max(1.0, (ex2 - ex1) * (ey2 - ey1))
					ioa_existing = inter / ex_area
					ioa_prop = inter / prop_area
					if ioa_existing > max_overlap or ioa_prop > max_overlap:
						safe = False
						break

			if safe:
				return prop_x, prop_y

		return random.randint(0, max_x), random.randint(0, max_y)

	def blend_and_paste(
		self,
		bg_img: np.ndarray,
		object_pixels: np.ndarray,
		object_mask: np.ndarray,
		top_left: tuple[int, int],
	) -> tuple[np.ndarray, tuple[int, int, int, int]]:
		x, y = top_left
		obj_h, obj_w = object_pixels.shape[:2]
		bg_h, bg_w = bg_img.shape[:2]
		if obj_h <= 0 or obj_w <= 0 or bg_h <= 0 or bg_w <= 0:
			return bg_img, (x, y, x, y)

		roi = bg_img[y : y + obj_h, x : x + obj_w]
		if roi.shape[:2] != (obj_h, obj_w):
			return bg_img, (x, y, x, y)

		mask = np.where(object_mask.astype(np.uint8) > 0, 255, 0).astype(np.uint8)
		if mask.max() == 0:
			return bg_img, (x, y, x, y)

		src = np.ascontiguousarray(object_pixels.astype(np.uint8))
		dst = np.ascontiguousarray(bg_img.astype(np.uint8))

		center_x = x + (obj_w // 2)
		center_y = y + (obj_h // 2)
		center_x = max(obj_w // 2, min(center_x, bg_w - ((obj_w + 1) // 2)))
		center_y = max(obj_h // 2, min(center_y, bg_h - ((obj_h + 1) // 2)))
		center = (int(center_x), int(center_y))

		try:
			bg_img = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
		except cv2.error:
			alpha = np.expand_dims(mask.astype(np.float32) / 255.0, axis=-1)
			blended = (src.astype(np.float32) * alpha) + (roi.astype(np.float32) * (1.0 - alpha))
			bg_img[y : y + obj_h, x : x + obj_w] = np.clip(blended, 0, 255).astype(np.uint8)

		return bg_img, (x, y, x + obj_w, y + obj_h)

	def execute_paste(
		self,
		bg_image_data: BackgroundImageData,
		objects_to_copy: list[MinedObject],
	) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
		background = cv2.imread(str(bg_image_data.image_path), cv2.IMREAD_COLOR)
		if background is None:
			raise RuntimeError(f"Failed to load background image: {bg_image_data.image_path}")

		image_h, image_w = background.shape[:2]
		new_boxes: list[tuple[int, float, float, float, float]] = []

		for obj in objects_to_copy:
			source = cv2.imread(str(obj.source_image_path), cv2.IMREAD_COLOR)
			if source is None:
				continue

			source_mask = self._load_source_mask(obj.source_image_path)
			class_id, xc, yc, w, h = obj.bbox
			x1, y1, x2, y2 = self._yolo_to_xyxy(xc, yc, w, h, source.shape[1], source.shape[0])
			if x2 <= x1 or y2 <= y1:
				continue

			object_pixels = source[y1:y2, x1:x2].copy()
			if object_pixels.size == 0:
				continue

			obj_h, obj_w = object_pixels.shape[:2]
			mask_crop = source_mask[y1:y2, x1:x2] if source_mask is not None else None
			object_mask = self._build_object_mask_from_crop(mask_crop, obj_h, obj_w)

			place_x, place_y = self.find_placement_coordinates(
				bg_image_data.existing_boxes,
				image_h,
				image_w,
				obj_h,
				obj_w,
			)

			background, bbox_xyxy = self.blend_and_paste(background, object_pixels, object_mask, (place_x, place_y))
			yolo_box = self._xyxy_to_yolo(*bbox_xyxy, image_w=image_w, image_h=image_h)
			new_boxes.append((class_id, *yolo_box))

		return background, new_boxes

	@staticmethod
	def _yolo_to_xyxy(x_center: float, y_center: float, width: float, height: float, image_w: int, image_h: int) -> tuple[int, int, int, int]:
		box_w = width * image_w
		box_h = height * image_h
		x1 = int((x_center * image_w) - box_w / 2)
		y1 = int((y_center * image_h) - box_h / 2)
		x2 = int(x1 + box_w)
		y2 = int(y1 + box_h)
		return max(x1, 0), max(y1, 0), min(x2, image_w), min(y2, image_h)

	@staticmethod
	def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, image_w: int, image_h: int) -> tuple[float, float, float, float]:
		width = max((x2 - x1) / image_w, 0.0)
		height = max((y2 - y1) / image_h, 0.0)
		x_center = ((x1 + x2) / 2) / image_w
		y_center = ((y1 + y2) / 2) / image_h
		return x_center, y_center, width, height

