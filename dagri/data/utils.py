import math
from pathlib import Path


def count_objects_in_label_file(label_file: Path) -> int:
    """
    Count number of objects in a YOLO label file.
    Each non-empty line is treated as one object annotation.
    """
    with open(label_file, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def compute_max_det_from_train_labels(
    train_labels_dir: str,
    percentile: float = 0.99,
    multiplier: int = 3,
) -> int:
    """
    Compute max_det from train label object counts.

    Rule: max_det = (percentile object-count per image) * multiplier.
    Percentile uses nearest-rank so 99% means 99% of images are <= selected count.
    """
    label_dir = Path(train_labels_dir)
    if not label_dir.exists() or not label_dir.is_dir():
        raise FileNotFoundError(f"Train labels directory not found: {label_dir}")

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found in: {label_dir}")

    object_counts = sorted(count_objects_in_label_file(p) for p in label_files)
    n = len(object_counts)
    rank_index = max(0, min(n - 1, math.ceil(percentile * n) - 1))
    percentile_count = object_counts[rank_index]

    return max(1, int(percentile_count * multiplier))
