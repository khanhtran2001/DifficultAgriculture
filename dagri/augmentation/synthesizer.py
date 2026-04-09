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
		blending_method: str = "seamless_clone",
		lab_gaussian_kernel_size: int = 15,
		rng: random.Random | None = None,
	):
		self.target_density = int(target_density)
		self.relative_multiplier = float(relative_multiplier)
		self.max_paste_per_image = int(max_paste_per_image)
		self.use_mask = bool(use_mask)
		self.segmentation_masks_dir = Path(segmentation_masks_dir).resolve() if segmentation_masks_dir else None
		self.blending_method = str(blending_method).lower().strip()
		if self.blending_method not in {"seamless_clone", "alpha", "none", "lab_gaussian"}:
			raise ValueError(
				"Unsupported blending_method="
				f"'{blending_method}'. Use 'seamless_clone', 'alpha', 'none', or 'lab_gaussian'."
			)
		kernel = int(lab_gaussian_kernel_size)
		if kernel < 1:
			kernel = 1
		if kernel % 2 == 0:
			kernel += 1
		self.lab_gaussian_kernel_size = kernel
		self.rng = rng or random.Random()
		self._source_mask_cache: dict[str, np.ndarray | None] = {}
		# Guardrails for strict Poisson mode to fail early with clear diagnostics.
		self._min_poisson_dim_px = 8
		self._min_poisson_mask_pixels = 16

	def get_mask_config_summary(self) -> str:
		if not self.use_mask:
			return "Mask usage: disabled (use_mask=False)"
		if self.segmentation_masks_dir is None:
			return "Mask usage: enabled but segmentation_masks_dir is not set"
		exists = self.segmentation_masks_dir.exists()
		return f"Mask usage: enabled, directory={self.segmentation_masks_dir} (exists={exists})"

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

	@staticmethod
	def _crop_to_mask_bounds(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		y_coords, x_coords = np.where(mask > 0)
		if x_coords.size == 0 or y_coords.size == 0:
			return image, mask

		x_min = int(x_coords.min())
		x_max = int(x_coords.max()) + 1
		y_min = int(y_coords.min())
		y_max = int(y_coords.max()) + 1
		return image[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]

	@staticmethod
	def _resize_pair(
		image: np.ndarray,
		mask: np.ndarray,
		scale_factor: float,
	) -> tuple[np.ndarray, np.ndarray]:
		scale_factor = float(scale_factor)
		if abs(scale_factor - 1.0) < 1e-6:
			return image, mask

		new_w = max(1, int(round(image.shape[1] * scale_factor)))
		new_h = max(1, int(round(image.shape[0] * scale_factor)))
		resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
		resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
		return resized_image, resized_mask

	@staticmethod
	def _rotate_pair(
		image: np.ndarray,
		mask: np.ndarray,
		rotation_deg: float,
	) -> tuple[np.ndarray, np.ndarray]:
		rotation_deg = float(rotation_deg)
		if abs(rotation_deg) < 1e-6:
			return image, mask

		h, w = image.shape[:2]
		center = (w / 2.0, h / 2.0)
		matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
		cos_abs = abs(matrix[0, 0])
		sin_abs = abs(matrix[0, 1])
		new_w = max(1, int(round((h * sin_abs) + (w * cos_abs))))
		new_h = max(1, int(round((h * cos_abs) + (w * sin_abs))))
		matrix[0, 2] += (new_w / 2.0) - center[0]
		matrix[1, 2] += (new_h / 2.0) - center[1]

		rotated_image = cv2.warpAffine(
			image,
			matrix,
			(new_w, new_h),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0),
		)
		rotated_mask = cv2.warpAffine(
			mask,
			matrix,
			(new_w, new_h),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		return rotated_image, rotated_mask

	def transform_object_patch(
		self,
		object_pixels: np.ndarray,
		object_mask: np.ndarray,
		scale_factor: float = 1.0,
		rotation_deg: float = 0.0,
	) -> tuple[np.ndarray, np.ndarray]:
		transformed_pixels, transformed_mask = self._resize_pair(object_pixels, object_mask, scale_factor)
		transformed_pixels, transformed_mask = self._rotate_pair(transformed_pixels, transformed_mask, rotation_deg)
		transformed_pixels, transformed_mask = self._crop_to_mask_bounds(transformed_pixels, transformed_mask)
		return transformed_pixels, transformed_mask

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
			return self.rng.randint(0, max_x), self.rng.randint(0, max_y)

		pixel_boxes = []
		for bbox in bg_existing_bboxes:
			_, xc, yc, w, h = bbox
			px_w, px_h = w * image_w, h * image_h
			x1 = (xc * image_w) - (px_w / 2)
			y1 = (yc * image_h) - (px_h / 2)
			pixel_boxes.append([x1, y1, x1 + px_w, y1 + px_h])

		for _ in range(max_attempts):
			tx1, ty1, tx2, ty2 = self.rng.choice(pixel_boxes)
			tw = tx2 - tx1
			th = ty2 - ty1
			jitter_x = int(max(tw * 1.5, obj_w * 2))
			jitter_y = int(max(th * 1.5, obj_h * 2))

			prop_x = int(self.rng.uniform(tx1 - jitter_x, tx2 + jitter_x))
			prop_y = int(self.rng.uniform(ty1 - jitter_y, ty2 + jitter_y))
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

		return self.rng.randint(0, max_x), self.rng.randint(0, max_y)

	def blend_and_paste(
		self,
		bg_img: np.ndarray,
		object_pixels: np.ndarray,
		object_mask: np.ndarray,
		top_left: tuple[int, int],
		debug_tag: str = "",
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

		if self.blending_method == "seamless_clone":
			mask_nonzero = int(np.count_nonzero(mask))
			if obj_h < self._min_poisson_dim_px or obj_w < self._min_poisson_dim_px:
				raise RuntimeError(
					"Poisson precheck failed: object crop is too small for stable seamlessClone. "
					f"debug={debug_tag}; obj_size=({obj_h},{obj_w}), "
					f"min_required={self._min_poisson_dim_px}px"
				)
			if mask_nonzero < self._min_poisson_mask_pixels:
				raise RuntimeError(
					"Poisson precheck failed: mask is too sparse for stable seamlessClone. "
					f"debug={debug_tag}; mask_nonzero={mask_nonzero}, "
					f"min_required={self._min_poisson_mask_pixels}"
				)

		src = np.ascontiguousarray(object_pixels.astype(np.uint8))
		dst = np.ascontiguousarray(bg_img.astype(np.uint8))

		center_x = x + (obj_w // 2)
		center_y = y + (obj_h // 2)
		center_x = max(obj_w // 2, min(center_x, bg_w - ((obj_w + 1) // 2)))
		center_y = max(obj_h // 2, min(center_y, bg_h - ((obj_h + 1) // 2)))
		center = (int(center_x), int(center_y))

		if self.blending_method == "none":
			# Hard paste with mask, no blending.
			mask_bool = mask > 0
			bg_roi = bg_img[y : y + obj_h, x : x + obj_w]
			bg_roi[mask_bool] = src[mask_bool]
			bg_img[y : y + obj_h, x : x + obj_w] = bg_roi
		elif self.blending_method == "lab_gaussian":
			corrected_src = self._lab_color_match(src, roi)
			blended = self._gaussian_blend(corrected_src, roi, mask, self.lab_gaussian_kernel_size)
			bg_img[y : y + obj_h, x : x + obj_w] = blended
		elif self.blending_method == "alpha":
			alpha = np.expand_dims(mask.astype(np.float32) / 255.0, axis=-1)
			blended = (src.astype(np.float32) * alpha) + (roi.astype(np.float32) * (1.0 - alpha))
			bg_img[y : y + obj_h, x : x + obj_w] = np.clip(blended, 0, 255).astype(np.uint8)
		else:
			self._validate_poisson_inputs(src, dst, mask, center, (x, y), debug_tag)
			try:
				bg_img = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
			except cv2.error as exc:
				non_zero = int(np.count_nonzero(mask))
				raise RuntimeError(
					"OpenCV seamlessClone failed with strict Poisson mode. "
					f"debug={debug_tag}; src_shape={src.shape}, dst_shape={dst.shape}, "
					f"mask_shape={mask.shape}, mask_nonzero={non_zero}, center={center}, top_left={(x, y)}; "
					f"opencv_error={exc}"
				) from exc

		return bg_img, (x, y, x + obj_w, y + obj_h)

	@staticmethod
	def _lab_color_match(source_crop: np.ndarray, target_region: np.ndarray) -> np.ndarray:
		source_lab = cv2.cvtColor(source_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
		target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)

		for channel in range(3):
			src_plane = source_lab[:, :, channel]
			tgt_plane = target_lab[:, :, channel]

			src_mean = float(np.mean(src_plane))
			src_std = float(np.std(src_plane))
			tgt_mean = float(np.mean(tgt_plane))
			tgt_std = float(np.std(tgt_plane))

			if src_std > 1e-6:
				source_lab[:, :, channel] = ((src_plane - src_mean) * (tgt_std / src_std)) + tgt_mean

		source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
		return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

	@staticmethod
	def _gaussian_blend(
		source_img: np.ndarray,
		target_img: np.ndarray,
		mask: np.ndarray,
		kernel_size: int,
	) -> np.ndarray:
		kernel = max(1, int(kernel_size))
		if kernel % 2 == 0:
			kernel += 1

		mask_f = mask.astype(np.float32) / 255.0
		smooth_alpha = cv2.GaussianBlur(mask_f, (kernel, kernel), 0)
		smooth_alpha = np.clip(smooth_alpha, 0.0, 1.0)
		smooth_alpha = np.expand_dims(smooth_alpha, axis=-1)

		blended = (source_img.astype(np.float32) * smooth_alpha) + (
			target_img.astype(np.float32) * (1.0 - smooth_alpha)
		)
		return np.clip(blended, 0, 255).astype(np.uint8)

	def _validate_poisson_inputs(
		self,
		src: np.ndarray,
		dst: np.ndarray,
		mask: np.ndarray,
		center: tuple[int, int],
		top_left: tuple[int, int],
		debug_tag: str,
	) -> None:
		x, y = top_left
		obj_h, obj_w = src.shape[:2]
		bg_h, bg_w = dst.shape[:2]
		mask_nonzero = int(np.count_nonzero(mask))

		if src.ndim != 3 or src.shape[2] != 3:
			raise RuntimeError(f"Invalid Poisson source image format: shape={src.shape}; debug={debug_tag}")
		if dst.ndim != 3 or dst.shape[2] != 3:
			raise RuntimeError(f"Invalid Poisson destination image format: shape={dst.shape}; debug={debug_tag}")
		if mask.ndim != 2:
			raise RuntimeError(f"Invalid Poisson mask format: shape={mask.shape}; debug={debug_tag}")
		if src.dtype != np.uint8 or dst.dtype != np.uint8 or mask.dtype != np.uint8:
			raise RuntimeError(
				f"Poisson requires uint8 inputs: src={src.dtype}, dst={dst.dtype}, mask={mask.dtype}; debug={debug_tag}"
			)
		if mask.shape != src.shape[:2]:
			raise RuntimeError(
				f"Mask/source size mismatch: mask={mask.shape}, src={src.shape[:2]}; debug={debug_tag}"
			)
		if obj_h < self._min_poisson_dim_px or obj_w < self._min_poisson_dim_px:
			raise RuntimeError(
				f"Poisson patch too small: src_shape={src.shape}, min_dim={self._min_poisson_dim_px}; debug={debug_tag}"
			)
		if mask_nonzero < self._min_poisson_mask_pixels:
			raise RuntimeError(
				f"Poisson mask too sparse: nonzero={mask_nonzero}, min_required={self._min_poisson_mask_pixels}; debug={debug_tag}"
			)

		cx, cy = center
		half_w = obj_w // 2
		half_h = obj_h // 2
		if (cx - half_w) < 0 or (cy - half_h) < 0 or (cx + (obj_w - half_w)) > bg_w or (cy + (obj_h - half_h)) > bg_h:
			raise RuntimeError(
				"Poisson center places patch outside destination bounds: "
				f"center={center}, src_shape={src.shape}, dst_shape={dst.shape}, top_left={(x, y)}; debug={debug_tag}"
			)

	def execute_paste(
		self,
		bg_image_data: BackgroundImageData,
		objects_to_copy: list[MinedObject],
		scale_factor: float = 1.0,
		rotation_deg: float = 0.0,
		min_transformed_area_ratio: float = 0.5,
		max_transformed_area_ratio: float = 2.0,
		min_transformed_side_px: int = 8,
		max_transformed_side_px: int | None = None,
	) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
		background = cv2.imread(str(bg_image_data.image_path), cv2.IMREAD_COLOR)
		if background is None:
			raise RuntimeError(f"Failed to load background image: {bg_image_data.image_path}")

		image_h, image_w = background.shape[:2]
		new_boxes: list[tuple[int, float, float, float, float]] = []
		occupied_boxes = list(bg_image_data.existing_boxes)

		for obj in objects_to_copy:
			source = cv2.imread(str(obj.source_image_path), cv2.IMREAD_COLOR)
			if source is None:
				continue

			class_id, xc, yc, w, h = obj.bbox
			x1, y1, x2, y2 = self._yolo_to_xyxy(xc, yc, w, h, source.shape[1], source.shape[0])
			if x2 <= x1 or y2 <= y1:
				continue

			object_pixels = source[y1:y2, x1:x2].copy()
			if object_pixels.size == 0:
				continue

			obj_h, obj_w = object_pixels.shape[:2]
			if obj_h <= 0 or obj_w <= 0:
				continue

			# Simple check: object should fit within image
			if obj_h >= image_h or obj_w >= image_w:
				continue

			place_x, place_y = self.find_placement_coordinates(
				occupied_boxes,
				image_h,
				image_w,
				obj_h,
				obj_w,
			)

			# Simple raw paste: copy pixels directly
			roi = background[place_y : place_y + obj_h, place_x : place_x + obj_w]
			if roi.shape[:2] == (obj_h, obj_w):
				background[place_y : place_y + obj_h, place_x : place_x + obj_w] = object_pixels

			bbox_xyxy = (place_x, place_y, place_x + obj_w, place_y + obj_h)
			yolo_box = self._xyxy_to_yolo(*bbox_xyxy, image_w=image_w, image_h=image_h)
			new_boxes.append((class_id, *yolo_box))
			occupied_boxes.append((class_id, *yolo_box))

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

