from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Literal

import numpy as np


WeightFunctionName = Literal["linear", "exponential", "power"]


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize score values into the [0, 1] range."""
    if not scores:
        return {}

    values = np.asarray(list(scores.values()), dtype=float)
    min_value = float(values.min())
    max_value = float(values.max())

    if np.isclose(min_value, max_value):
        return {name: 1.0 for name in scores}

    scale = max_value - min_value
    return {
        name: float((float(value) - min_value) / scale)
        for name, value in scores.items()
    }


def linear_weight(normalized_scores: Dict[str, float], gamma: float = 1.0) -> Dict[str, float]:
    """Linear weighting function: w(i) = 1 + gamma * S_norm(i)."""
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")

    return {name: float(1.0 + gamma * score) for name, score in normalized_scores.items()}


def exponential_weight(normalized_scores: Dict[str, float], gamma: float = 1.0) -> Dict[str, float]:
    """Exponential weighting function: w(i) = exp(gamma * S_norm(i))."""
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")

    return {name: float(np.exp(gamma * score)) for name, score in normalized_scores.items()}


def power_weight(normalized_scores: Dict[str, float], gamma: float = 1.0) -> Dict[str, float]:
    """Power weighting function: w(i) = S_norm(i)^gamma + 1."""
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    return {name: float((score ** gamma) + 1.0) for name, score in normalized_scores.items()}


def get_weight_function(function_name: WeightFunctionName) -> Callable[[Dict[str, float], float], Dict[str, float]]:
    """Return a weight function by name."""
    functions: Dict[str, Callable[[Dict[str, float], float], Dict[str, float]]] = {
        "linear": linear_weight,
        "exponential": exponential_weight,
        "power": power_weight,
    }

    if function_name not in functions:
        raise ValueError(f"Unknown weight function: {function_name}. Choose from {list(functions)}")

    return functions[function_name]


def scores_to_weight_map(
    scores: Dict[str, float],
    function_name: WeightFunctionName = "linear",
    gamma: float = 1.0,
    normalize: bool = True,
) -> Dict[str, float]:
    """Convert raw difficulty scores to sampling weights."""
    resolved_scores = normalize_scores(scores) if normalize else dict(scores)
    weight_function = get_weight_function(function_name)
    return weight_function(resolved_scores, gamma=gamma)


def scoring_results_to_score_map(scoring_results: Any, key: str = "name") -> Dict[str, float]:
    """Extract an image score map from ScoringResults or a compatible payload."""
    image_scores: Dict[str, float] = {}

    if isinstance(scoring_results, dict):
        image_difficulties = scoring_results.get("image_difficulties", []) or []
    else:
        image_difficulties = getattr(scoring_results, "image_difficulties", []) or []

    for image_difficulty in image_difficulties:
        if isinstance(image_difficulty, dict):
            image_path = Path(str(image_difficulty.get("image_path", "")))
            difficulty_score = float(image_difficulty.get("difficulty_score", 0.0))
        else:
            image_path = Path(getattr(image_difficulty, "image_path", ""))
            difficulty_score = float(getattr(image_difficulty, "difficulty_score", 0.0))

        if key == "stem":
            image_key = image_path.stem
        elif key == "path":
            image_key = str(image_path)
        else:
            image_key = image_path.name
        image_scores[image_key] = difficulty_score

    return image_scores


def weight_summary(weights: Dict[str, float]) -> Dict[str, float]:
    """Return summary statistics for a weight dictionary."""
    if not weights:
        return {}

    values = np.asarray(list(weights.values()), dtype=float)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "range": float(values.max() - values.min()),
        "num_images": int(len(weights)),
    }
