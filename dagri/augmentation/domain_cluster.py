from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class DomainClusterer:
	def __init__(self, images_dir: str, image_extensions: list[str] | None = None):
		self.images_dir = Path(images_dir)
		self.image_extensions = {
			ext.lower()
			for ext in (image_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
		}

	def _collect_image_paths(self) -> list[Path]:
		if not self.images_dir.exists():
			return []
		return sorted(
			[
				path
				for path in self.images_dir.rglob("*")
				if path.is_file() and path.suffix.lower() in self.image_extensions
			]
		)

	@staticmethod
	def _extract_feature(image_path: Path) -> np.ndarray:
		image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
		if image is None:
			return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mean_r, mean_g, mean_b = rgb.mean(axis=(0, 1))
		brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
		return np.array([mean_r, mean_g, mean_b, brightness], dtype=np.float32)

	def extract_visual_domains(self, auto_k: bool = True, max_k: int = 8) -> dict[str, int]:
		image_paths = self._collect_image_paths()
		if not image_paths:
			return {}

		if len(image_paths) == 1:
			return {image_paths[0].name: 0}

		features = np.vstack([self._extract_feature(path) for path in image_paths])
		n_samples = len(image_paths)

		if auto_k:
			upper_k = min(max_k, n_samples - 1)
			best_k = 2
			best_score = -1.0
			for k in range(2, upper_k + 1):
				labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(features)
				score = silhouette_score(features, labels)
				if score > best_score:
					best_score = score
					best_k = k
		else:
			best_k = min(max_k, max(2, int(np.sqrt(n_samples))))

		labels = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit_predict(features)
		return {image_path.name: int(label) for image_path, label in zip(image_paths, labels)}

