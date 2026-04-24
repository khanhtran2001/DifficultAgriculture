from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dagri.augmentation.augumentor import CopyPasteAugmentor, ObjectData


@dataclass
class SelectionStats:
    label: str
    mean_score: float
    mean_area: float
    total_selected: int


def _load_objects_from_score_results(score_results_path: Path) -> list[ObjectData]:
    with score_results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    objects: list[ObjectData] = []
    for image_entry in payload.get("image_difficulties", []):
        image_path = Path(image_entry["image_path"])
        image_name = image_path.name

        for object_entry in image_entry.get("objects_score", []):
            bbox = object_entry.get("bounding_box") or {}
            width = float(bbox.get("width") or 0.0)
            height = float(bbox.get("height") or 0.0)
            if width <= 0.0 or height <= 0.0:
                continue

            objects.append(
                ObjectData(
                    image_name=image_name,
                    image_path=image_path,
                    object_index=int(object_entry["object_id"]),
                    bbox=(
                        int(object_entry.get("class_id", 0)),
                        float(bbox.get("x_center") or 0.0),
                        float(bbox.get("y_center") or 0.0),
                        width,
                        height,
                    ),
                    score=float(object_entry["difficulty_score"]),
                )
            )

    return objects


def _compute_stats(
    label: str,
    objects: list[ObjectData],
    *,
    use_score_guidance: bool,
    score_weight_function: str,
    score_alpha: float,
    trials: int,
    num_to_select: int,
    seed: int,
) -> SelectionStats:
    augmentor = CopyPasteAugmentor(config={})
    rng = random.Random(seed)

    sum_score = 0.0
    sum_area = 0.0
    total_selected = 0

    for _ in range(trials):
        selected = augmentor._select_objects(
            objects=objects,
            reuse_counts={},
            max_reuse=None,
            num_to_select=num_to_select,
            use_score=use_score_guidance,
            score_weight_function=score_weight_function,
            score_alpha=score_alpha,
            rng=rng,
        )

        total_selected += len(selected)
        for obj in selected:
            sum_score += float(obj.score)
            sum_area += float(obj.bbox[3]) * float(obj.bbox[4])

    if total_selected == 0:
        raise RuntimeError("No objects were selected during sampling")

    return SelectionStats(
        label=label,
        mean_score=sum_score / total_selected,
        mean_area=sum_area / total_selected,
        total_selected=total_selected,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that use_score_guidance=true prefers higher-score objects by "
            "comparing sampled mean score and mean area against random selection."
        )
    )
    parser.add_argument(
        "--score-results",
        type=Path,
        required=True,
        help="Path to score_results.json",
    )
    parser.add_argument("--trials", type=int, default=2000, help="Sampling trials")
    parser.add_argument("--num-to-select", type=int, default=8, help="Objects selected per trial")
    parser.add_argument("--weight-function", type=str, default="linear", help="linear or exponential")
    parser.add_argument("--alpha", type=float, default=3.0, help="Score weighting alpha")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    objects = _load_objects_from_score_results(args.score_results)
    if not objects:
        raise RuntimeError("No valid objects found in score_results file")

    pool_mean_score = sum(float(obj.score) for obj in objects) / len(objects)
    pool_mean_area = sum(float(obj.bbox[3]) * float(obj.bbox[4]) for obj in objects) / len(objects)

    random_stats = _compute_stats(
        "random (use_score_guidance=false)",
        objects,
        use_score_guidance=False,
        score_weight_function=args.weight_function,
        score_alpha=args.alpha,
        trials=args.trials,
        num_to_select=args.num_to_select,
        seed=args.seed,
    )
    guided_stats = _compute_stats(
        "score-guided (use_score_guidance=true)",
        objects,
        use_score_guidance=True,
        score_weight_function=args.weight_function,
        score_alpha=args.alpha,
        trials=args.trials,
        num_to_select=args.num_to_select,
        seed=args.seed,
    )

    print(f"Object pool count: {len(objects)}")
    print(f"Pool mean score: {pool_mean_score:.6f}")
    print(f"Pool mean area:  {pool_mean_area:.8f}")
    print()
    print(f"{random_stats.label}")
    print(f"  selected objects: {random_stats.total_selected}")
    print(f"  mean score:       {random_stats.mean_score:.6f}")
    print(f"  mean area:        {random_stats.mean_area:.8f}")
    print()
    print(f"{guided_stats.label}")
    print(f"  selected objects: {guided_stats.total_selected}")
    print(f"  mean score:       {guided_stats.mean_score:.6f}")
    print(f"  mean area:        {guided_stats.mean_area:.8f}")
    print()
    print(f"delta mean score (guided-random): {guided_stats.mean_score - random_stats.mean_score:.6f}")
    print(f"delta mean area  (guided-random): {guided_stats.mean_area - random_stats.mean_area:.8f}")

    if guided_stats.mean_score <= random_stats.mean_score:
        print("FAIL: use_score_guidance=true did not increase mean selected score")
        return 1

    print("PASS: use_score_guidance=true selects higher-score objects on average")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
