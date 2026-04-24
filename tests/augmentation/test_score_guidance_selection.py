from __future__ import annotations

import math
import random
import unittest
from pathlib import Path

from dagri.augmentation.augumentor import CopyPasteAugmentor, ObjectData


class TestScoreGuidanceSelection(unittest.TestCase):
    @staticmethod
    def _make_objects(num_objects: int = 20) -> list[ObjectData]:
        objects: list[ObjectData] = []
        for idx in range(num_objects):
            score = idx / (num_objects - 1)
            # Keep area strongly correlated with score for an easy-to-interpret check.
            area = 0.001 + (0.04 * score)
            side = math.sqrt(area)
            objects.append(
                ObjectData(
                    image_name=f"img_{idx // 2}.jpg",
                    image_path=Path(f"/tmp/img_{idx // 2}.jpg"),
                    object_index=idx,
                    bbox=(0, 0.5, 0.5, side, side),
                    score=score,
                )
            )
        return objects

    @staticmethod
    def _sample_means(
        objects: list[ObjectData],
        *,
        use_score_guidance: bool,
        trials: int,
        num_to_select: int,
        alpha: float,
        seed: int,
    ) -> tuple[float, float]:
        augmentor = CopyPasteAugmentor(config={})
        rng = random.Random(seed)

        total_score = 0.0
        total_area = 0.0
        total_count = 0

        for _ in range(trials):
            selected = augmentor._select_objects(
                objects=objects,
                reuse_counts={},
                max_reuse=None,
                num_to_select=num_to_select,
                use_score=use_score_guidance,
                score_weight_function="linear",
                score_alpha=alpha,
                rng=rng,
            )

            total_count += len(selected)
            for obj in selected:
                total_score += float(obj.score)
                total_area += float(obj.bbox[3]) * float(obj.bbox[4])

        self_mean_score = total_score / total_count
        self_mean_area = total_area / total_count
        return self_mean_score, self_mean_area

    def test_use_score_guidance_prefers_high_score_objects(self) -> None:
        objects = self._make_objects()

        random_mean_score, random_mean_area = self._sample_means(
            objects,
            use_score_guidance=False,
            trials=3000,
            num_to_select=5,
            alpha=3.0,
            seed=123,
        )
        guided_mean_score, guided_mean_area = self._sample_means(
            objects,
            use_score_guidance=True,
            trials=3000,
            num_to_select=5,
            alpha=3.0,
            seed=123,
        )

        print(f"random mean score={random_mean_score:.6f}, mean area={random_mean_area:.8f}")
        print(f"guided mean score={guided_mean_score:.6f}, mean area={guided_mean_area:.8f}")

        self.assertGreater(guided_mean_score, random_mean_score)
        self.assertGreater(random_mean_area, 0.0)
        self.assertGreater(guided_mean_area, 0.0)


if __name__ == "__main__":
    unittest.main()
