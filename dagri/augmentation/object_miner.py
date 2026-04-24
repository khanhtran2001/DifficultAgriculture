from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from dagri.interfaces import ScoringResults


@dataclass
class BackgroundImageData:
	image_name: str
	image_path: Path
	existing_boxes: list[tuple[int, float, float, float, float]]
	simg_score: float


@dataclass
class MinedObject:
	source_image_name: str
	source_image_path: Path
	object_index: int
	bbox: tuple[int, float, float, float, float]
	area_px: float
	sobj_score: float


@dataclass
class MiningRequest:
	use_image_score: bool
	use_object_score: bool
	paste_relative_multiplier: float
	target_density: int | None = None
	max_paste_per_image: int | None = None
	max_background_reuse: int | None = None
	max_object_reuse: int | None = None


@dataclass
class MiningSelection:
	background: BackgroundImageData
	objects_to_copy: list[MinedObject]
	target_count: int


class ObjectMiner:
	def __init__(
		self,
		images_dir: str,
		labels_dir: str,
		scoring_results: ScoringResults,
		top_object_fraction: float | None,
		object_noise_cap: float | None,
		image_weight_mode: str,
		object_weight_mode: str,
		weight_scale: float,
		max_object_area_px: float | None,
		image_extensions: list[str],
		rng: random.Random,
	):
		self.images_dir = Path(images_dir)
		self.labels_dir = Path(labels_dir)
		self.scoring_results = scoring_results
		self.top_object_fraction = None if top_object_fraction is None else float(top_object_fraction)
		self.object_noise_cap = None if object_noise_cap is None else float(object_noise_cap)
		self.image_weight_mode = str(image_weight_mode).lower()
		self.object_weight_mode = str(object_weight_mode).lower()
		self.weight_scale = float(weight_scale)
		self.max_object_area_px = None if max_object_area_px is None else float(max_object_area_px)
		self.image_extensions = {ext.lower() for ext in image_extensions}
		self.rng = rng

		self.background_pool: list[BackgroundImageData] = []
		self.object_pool: list[MinedObject] = []
		self.objects_by_image: dict[str, list[MinedObject]] = {}
		self.total_images: int = 0

		self._image_score_map: dict[str, float] = {}
		self._object_score_map: dict[str, float] = {}
		self._build_score_maps()

	def _build_score_maps(self) -> None:
		for img in self.scoring_results.image_difficulties:
			p = Path(img.image_path)
			self._image_score_map[p.name] = float(img.difficulty_score)
			self._image_score_map[p.stem] = float(img.difficulty_score)
			for obj in img.objects_score:
				self._object_score_map[f"{p.name}:{int(obj.object_id)}"] = float(obj.difficulty_score)
				self._object_score_map[f"{p.stem}:{int(obj.object_id)}"] = float(obj.difficulty_score)

	def load_data(self) -> None:
		self.background_pool = []
		self.object_pool = []
		self.objects_by_image = {}

		image_paths = sorted(
			[
				path
				for path in self.images_dir.rglob("*")
				if path.is_file() and path.suffix.lower() in self.image_extensions
			]
		)

		for image_path in image_paths:
			image_name = image_path.name
			label_path = self.labels_dir / f"{image_path.stem}.txt"
			boxes = self._read_yolo_labels(label_path)
			image_w, image_h = self._read_image_size(image_path)

			bg_data = BackgroundImageData(
				image_name=image_name,
				image_path=image_path,
				existing_boxes=boxes,
				simg_score=self._get_image_score(image_name),
			)
			self.background_pool.append(bg_data)
			self.objects_by_image[image_name] = []

			for index, box in enumerate(boxes):
				area_px = self._bbox_area_px(box, image_w, image_h)
				if self.max_object_area_px is not None and area_px >= self.max_object_area_px:
					continue
				obj = MinedObject(
					source_image_name=image_name,
					source_image_path=image_path,
					object_index=index,
					bbox=box,
					area_px=area_px,
					sobj_score=self._get_object_score(image_name, index),
				)
				self.object_pool.append(obj)
				self.objects_by_image[image_name].append(obj)

		self.total_images = len(self.background_pool)

	def _get_image_score(self, image_name: str) -> float:
		stem = Path(image_name).stem
		if image_name in self._image_score_map:
			return float(self._image_score_map[image_name])
		if stem in self._image_score_map:
			return float(self._image_score_map[stem])
		raise ValueError(f"Missing image difficulty score for '{image_name}'")

	def _get_object_score(self, image_name: str, index: int) -> float:
		stem = Path(image_name).stem
		key = f"{image_name}:{index}"
		stem_key = f"{stem}:{index}"
		if key in self._object_score_map:
			return float(self._object_score_map[key])
		if stem_key in self._object_score_map:
			return float(self._object_score_map[stem_key])
		raise ValueError(f"Missing object difficulty score for '{image_name}' object index {index}")

	@staticmethod
	def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
		if not label_path.exists():
			return []

		boxes: list[tuple[int, float, float, float, float]] = []
		with label_path.open("r", encoding="utf-8") as handle:
			for line in handle:
				parts = line.strip().split()
				if len(parts) != 5:
					continue
				class_id = int(float(parts[0]))
				x_center, y_center, width, height = map(float, parts[1:])
				boxes.append((class_id, x_center, y_center, width, height))
		return boxes

	@staticmethod
	def _read_image_size(image_path: Path) -> tuple[int, int]:
		with Image.open(image_path) as image:
			return int(image.width), int(image.height)

	@staticmethod
	def _bbox_area_px(bbox: tuple[int, float, float, float, float], image_w: int, image_h: int) -> float:
		_, _, _, bw, bh = bbox
		return float(max(bw, 0.0) * max(bh, 0.0) * image_w * image_h)

	@staticmethod
	def _object_reuse_key(image_name: str, object_index: int) -> str:
		return f"{image_name}:{int(object_index)}"

	def get_available_backgrounds(
		self,
		background_reuse_counts: dict[str, int],
		max_background_reuse: int | None,
		excluded_names: set[str] | None = None,
	) -> list[BackgroundImageData]:
		excluded_names = excluded_names or set()
		return [
			bg
			for bg in self.background_pool
			if bg.image_name not in excluded_names
			and (
				max_background_reuse is None
				or background_reuse_counts.get(bg.image_name, 0) < max_background_reuse
			)
		]

	def select_background_image(
		self,
		background_reuse_counts: dict[str, int],
		max_background_reuse: int | None,
		use_image_score: bool,
		excluded_names: set[str] | None = None,
	) -> BackgroundImageData | None:
		available = self.get_available_backgrounds(
			background_reuse_counts=background_reuse_counts,
			max_background_reuse=max_background_reuse,
			excluded_names=excluded_names,
		)
		if not available:
			return None
		if not use_image_score:
			return self.rng.choice(available)
		weights = self._build_weights([bg.simg_score for bg in available], self.image_weight_mode)
		return self.rng.choices(available, weights=weights, k=1)[0]

	def get_available_objects_for_image(
		self,
		image_name: str,
		object_reuse_counts: dict[str, int],
		max_object_reuse: int | None,
		use_object_score: bool,
	) -> list[MinedObject]:
		objects = list(self.objects_by_image.get(image_name, []))
		if max_object_reuse is not None:
			objects = [
				obj
				for obj in objects
				if object_reuse_counts.get(self._object_reuse_key(obj.source_image_name, obj.object_index), 0)
				< max_object_reuse
			]
		if not objects:
			return []
		if not use_object_score:
			return objects

		filtered = objects
		if self.object_noise_cap is not None:
			filtered = [obj for obj in filtered if obj.sobj_score < self.object_noise_cap]
		if not filtered:
			return []
		if self.top_object_fraction is None:
			return filtered

		filtered.sort(key=lambda item: item.sobj_score, reverse=True)
		fraction = max(0.0, min(self.top_object_fraction, 1.0))
		keep_count = max(1, int(len(filtered) * fraction))
		return filtered[:keep_count]

	def select_objects_to_copy(
		self,
		compatible_pool: list[MinedObject],
		target_count: int,
		use_object_score: bool,
	) -> list[MinedObject]:
		if target_count <= 0 or not compatible_pool:
			return []
		count = min(target_count, len(compatible_pool))
		if not use_object_score:
			return self.rng.sample(compatible_pool, count)
		return self._weighted_sample_without_replacement(compatible_pool, count, self.object_weight_mode)

	def calculate_target_paste_count(
		self,
		current_objects: int,
		request: MiningRequest,
	) -> int:
		relative_limit = int(max(float(current_objects) * float(request.paste_relative_multiplier), 1.0))
		caps = [relative_limit]
		if request.target_density is not None:
			caps.append(max(int(request.target_density) - int(current_objects), 0))
		if request.max_paste_per_image is not None:
			caps.append(max(int(request.max_paste_per_image), 0))
		return max(0, min(caps))

	def mine_selection(
		self,
		background_reuse_counts: dict[str, int],
		object_reuse_counts: dict[str, int],
		request: MiningRequest,
		max_background_pick_attempts: int,
	) -> MiningSelection | None:
		attempted_backgrounds: set[str] = set()
		for _ in range(max_background_pick_attempts):
			bg_candidate = self.select_background_image(
				background_reuse_counts=background_reuse_counts,
				max_background_reuse=request.max_background_reuse,
				use_image_score=request.use_image_score,
				excluded_names=attempted_backgrounds,
			)
			if bg_candidate is None:
				return None

			attempted_backgrounds.add(bg_candidate.image_name)
			target_count = self.calculate_target_paste_count(
				current_objects=len(bg_candidate.existing_boxes),
				request=request,
			)
			if target_count <= 0:
				continue

			compatible = self.get_available_objects_for_image(
				image_name=bg_candidate.image_name,
				object_reuse_counts=object_reuse_counts,
				max_object_reuse=request.max_object_reuse,
				use_object_score=request.use_object_score,
			)
			candidate_objects = self.select_objects_to_copy(
				compatible_pool=compatible,
				target_count=target_count,
				use_object_score=request.use_object_score,
			)
			if not candidate_objects:
				continue

			return MiningSelection(
				background=bg_candidate,
				objects_to_copy=candidate_objects,
				target_count=target_count,
			)

		return None

	def _score_to_weight(self, score: float, mode: str) -> float:
		safe_score = float(score)
		if mode == "exponential":
			exponent = max(min(safe_score * self.weight_scale, 50.0), -50.0)
			return max(math.exp(exponent), 1e-6)
		return max(safe_score, 1e-6)

	def _build_weights(self, scores: list[float], mode: str) -> list[float]:
		return [self._score_to_weight(score, mode) for score in scores]

	def _weighted_sample_without_replacement(self, items: list[MinedObject], k: int, weight_mode: str) -> list[MinedObject]:
		remaining = list(items)
		selected: list[MinedObject] = []
		for _ in range(k):
			if not remaining:
				break
			weights = self._build_weights([item.sobj_score for item in remaining], weight_mode)
			picked = self.rng.choices(remaining, weights=weights, k=1)[0]
			selected.append(picked)
			remaining.remove(picked)
		return selected

