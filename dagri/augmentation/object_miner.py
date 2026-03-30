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
	domain_id: int
	simg_score: float


@dataclass
class MinedObject:
	source_image_name: str
	source_image_path: Path
	object_index: int
	bbox: tuple[int, float, float, float, float]
	area_px: float
	domain_id: int
	sobj_score: float


class ObjectMiner:
	def __init__(
		self,
		images_dir: str,
		labels_dir: str,
		scoring_results: ScoringResults,
		scoring_mode: str = "score_targeted",
		top_object_fraction: float = 0.3,
		object_noise_cap: float = 100.0,
		background_weight_mode: str = "linear",
		object_weight_mode: str = "linear",
		weight_scale: float = 3.0,
		max_object_area_px: float = 1024.0,
		image_extensions: list[str] | None = None,
	):
		self.images_dir = Path(images_dir)
		self.labels_dir = Path(labels_dir)
		self.scoring_results = scoring_results
		self.scoring_mode = str(scoring_mode).lower()
		self.top_object_fraction = float(top_object_fraction)
		self.object_noise_cap = float(object_noise_cap)
		self.background_weight_mode = str(background_weight_mode).lower()
		self.object_weight_mode = str(object_weight_mode).lower()
		self.weight_scale = float(weight_scale)
		self.max_object_area_px = float(max_object_area_px)
		self.image_extensions = {
			ext.lower() for ext in (image_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
		}

		self.background_pool: list[BackgroundImageData] = []
		self.object_pool: list[MinedObject] = []
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

	def load_data(self, domain_map: dict[str, int]) -> None:
		image_paths = sorted(
			[
				path
				for path in self.images_dir.rglob("*")
				if path.is_file() and path.suffix.lower() in self.image_extensions
			]
		)

		for image_path in image_paths:
			image_name = image_path.name
			domain_id = int(domain_map.get(image_name, -1))
			label_path = self.labels_dir / f"{image_path.stem}.txt"
			boxes = self._read_yolo_labels(label_path)
			image_w, image_h = self._read_image_size(image_path)

			bg_data = BackgroundImageData(
				image_name=image_name,
				image_path=image_path,
				existing_boxes=boxes,
				domain_id=domain_id,
				simg_score=self._get_image_score(image_name),
			)
			self.background_pool.append(bg_data)

			for index, box in enumerate(boxes):
				area_px = self._bbox_area_px(box, image_w, image_h)
				self.object_pool.append(
					MinedObject(
						source_image_name=image_name,
						source_image_path=image_path,
						object_index=index,
						bbox=box,
						area_px=area_px,
						domain_id=domain_id,
						sobj_score=self._get_object_score(image_name, index),
					)
				)

		self.total_images = len(self.background_pool)

	def _get_image_score(self, image_name: str) -> float:
		stem = Path(image_name).stem
		return float(self._image_score_map.get(image_name, self._image_score_map.get(stem, 1.0)))

	def _get_object_score(self, image_name: str, index: int) -> float:
		stem = Path(image_name).stem
		return float(
			self._object_score_map.get(f"{image_name}:{index}", self._object_score_map.get(f"{stem}:{index}", 1.0))
		)

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

	def select_background_image(self) -> BackgroundImageData:
		if not self.background_pool:
			raise RuntimeError("Background pool is empty. Did you call load_data()?")
		if self.scoring_mode == "random":
			return random.choice(self.background_pool)
		weights = self._build_weights([bg.simg_score for bg in self.background_pool], self.background_weight_mode)
		return random.choices(self.background_pool, weights=weights, k=1)[0]

	def get_compatible_objects(self, bg_image: BackgroundImageData) -> list[MinedObject]:
		compatible = [
			obj
			for obj in self.object_pool
			if obj.domain_id == bg_image.domain_id and obj.area_px < self.max_object_area_px
		]
		if not compatible or self.scoring_mode == "random":
			return compatible

		filtered = [obj for obj in compatible if obj.sobj_score < self.object_noise_cap]
		if not filtered:
			return []
		filtered.sort(key=lambda item: item.sobj_score, reverse=True)
		keep_count = max(1, int(len(filtered) * self.top_object_fraction))
		return filtered[:keep_count]

	def select_objects_to_copy(self, compatible_pool: list[MinedObject], target_count: int) -> list[MinedObject]:
		if target_count <= 0 or not compatible_pool:
			return []
		count = min(target_count, len(compatible_pool))
		if self.scoring_mode == "random":
			return random.sample(compatible_pool, count)
		return self._weighted_sample_without_replacement(compatible_pool, count, self.object_weight_mode)

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
			picked = random.choices(remaining, weights=weights, k=1)[0]
			selected.append(picked)
			remaining.remove(picked)
		return selected

