import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import yaml

# Allow running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dagri.general.config_manager import ConfigManager
from dagri.general.result_manager import ResultManager
from dagri.interfaces import DatasetProperties, ScoringResults, ImageDifficultyProperties, ObjectDifficultyProperties
from dagri.baseline import Baseline
from dagri.augmentation import CopyPasteAugmentor

import experiments.utils as exputils


RESULTS_DIR = Path(f"results/{Path(__file__).stem}")
CONFIG_DIR = Path("/home/khanh/Projects/DifficultyAgri/configs/experiments/minneapple_yolo.yaml")
CACHE_ROOT = Path("/home/khanh/Projects/DifficultyAgri/.cache_result/no_trad_aug/minneapple")
DEFAULT_ALPHAS = [1.0, 3.0, 5.0]


def _metric_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean_val = float(statistics.mean(values))
    std_val = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean_val, "std": std_val}


def _to_eval_dict(evaluation_results) -> dict:
    return {
        "COCO_AP": float(evaluation_results.COCO_AP),
        "COCO_AP50": float(evaluation_results.COCO_AP50),
        "COCO_AP75": float(evaluation_results.COCO_AP75),
        "AP_small": float(evaluation_results.AP_small),
        "AP_medium": float(evaluation_results.AP_medium),
        "AP_large": float(evaluation_results.AP_large),
    }


def _delta_eval_dict(variant_eval: dict, baseline_eval: dict) -> dict:
    return {
        "COCO_AP": float(variant_eval["COCO_AP"] - baseline_eval["COCO_AP"]),
        "COCO_AP50": float(variant_eval["COCO_AP50"] - baseline_eval["COCO_AP50"]),
        "COCO_AP75": float(variant_eval["COCO_AP75"] - baseline_eval["COCO_AP75"]),
        "AP_small": float(variant_eval["AP_small"] - baseline_eval["AP_small"]),
        "AP_medium": float(variant_eval["AP_medium"] - baseline_eval["AP_medium"]),
        "AP_large": float(variant_eval["AP_large"] - baseline_eval["AP_large"]),
    }


def _aggregate_condition(condition_runs: list[dict]) -> dict:
    return {
        "augmented_eval_on_original_test": {
            "COCO_AP": _metric_stats([r["augmented_eval_on_original_test"]["COCO_AP"] for r in condition_runs]),
            "COCO_AP50": _metric_stats([r["augmented_eval_on_original_test"]["COCO_AP50"] for r in condition_runs]),
            "COCO_AP75": _metric_stats([r["augmented_eval_on_original_test"]["COCO_AP75"] for r in condition_runs]),
            "AP_small": _metric_stats([r["augmented_eval_on_original_test"]["AP_small"] for r in condition_runs]),
            "AP_medium": _metric_stats([r["augmented_eval_on_original_test"]["AP_medium"] for r in condition_runs]),
            "AP_large": _metric_stats([r["augmented_eval_on_original_test"]["AP_large"] for r in condition_runs]),
        },
        "delta_aug_minus_baseline": {
            "COCO_AP": _metric_stats([r["delta_aug_minus_baseline"]["COCO_AP"] for r in condition_runs]),
            "COCO_AP50": _metric_stats([r["delta_aug_minus_baseline"]["COCO_AP50"] for r in condition_runs]),
            "COCO_AP75": _metric_stats([r["delta_aug_minus_baseline"]["COCO_AP75"] for r in condition_runs]),
            "AP_small": _metric_stats([r["delta_aug_minus_baseline"]["AP_small"] for r in condition_runs]),
            "AP_medium": _metric_stats([r["delta_aug_minus_baseline"]["AP_medium"] for r in condition_runs]),
            "AP_large": _metric_stats([r["delta_aug_minus_baseline"]["AP_large"] for r in condition_runs]),
        },
    }


def _resolve_three_seeds(config_path: str, override_seeds: str | None) -> list[int]:
    if override_seeds:
        parsed = [int(x.strip()) for x in override_seeds.split(",") if x.strip()]
        if len(parsed) != 3:
            raise ValueError("--seeds must contain exactly 3 comma-separated integers, e.g. 123,456,789")
        return parsed

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    random_seeds = (cfg.get("general") or {}).get("random_seed", [123, 456, 789])
    if isinstance(random_seeds, int):
        random_seeds = [int(random_seeds)]
    else:
        random_seeds = [int(s) for s in random_seeds]

    if len(random_seeds) < 3:
        raise ValueError(
            "Need at least 3 seeds in general.random_seed (or pass --seeds with exactly 3 values)."
        )
    return random_seeds[:3]


def _parse_alphas(alpha_str: str | None) -> list[float]:
    if not alpha_str:
        return DEFAULT_ALPHAS
    values = [float(x.strip()) for x in alpha_str.split(",") if x.strip()]
    if not values:
        raise ValueError("--alphas must contain at least one numeric value.")
    return values


def _alpha_key(alpha: float) -> str:
    alpha_txt = str(alpha).replace(".", "p")
    return f"alpha_{alpha_txt}"


def _load_dataset_properties(path: Path) -> DatasetProperties:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetProperties(**data)


def _load_scoring_results(path: Path) -> ScoringResults:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_difficulties = []
    for img_data in data.get("image_difficulties", []):
        objects_score = [ObjectDifficultyProperties(**obj) for obj in img_data.get("objects_score", [])]
        img_data["objects_score"] = objects_score
        image_difficulties.append(ImageDifficultyProperties(**img_data))

    data["image_difficulties"] = image_difficulties
    return ScoringResults(**data)


def run_experiment(config_path: str, seeds: list[int], alphas: list[float]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frozen_config_path = RESULTS_DIR / "frozen_config.yaml"
    exputils.copy_yaml_config(config_path, frozen_config_path)

    config_manager = ConfigManager()
    config_manager.load_all_configs(config_path)

    baseline_model_config_template = config_manager.baseline_config
    augmentation_config_template = dict(config_manager.augmentation_config or {})

    result_manager = ResultManager()

    dataset_properties_path = CACHE_ROOT / "dataset_properties.json"
    if not dataset_properties_path.exists():
        raise FileNotFoundError(f"Missing cached dataset properties: {dataset_properties_path}")
    initial_dataset_properties = _load_dataset_properties(dataset_properties_path)

    alpha_keys = [_alpha_key(alpha) for alpha in alphas]

    print(f"Running alpha sweep with 3 seeds: {seeds}")
    print(f"Alpha values: {alphas}")
    print(f"Using cached step 1-3 artifacts from: {CACHE_ROOT}")

    summary: dict = {
        "seeds": seeds,
        "alphas": alphas,
        "conditions": {k: [] for k in alpha_keys},
        "baseline": [],
        "aggregate": {},
        "ranking_by_delta_coco_ap": [],
    }

    for seed in seeds:
        print(f"\n========== Seed {seed} ==========")
        seed_root = RESULTS_DIR / f"seed_{seed}"
        step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, _ = exputils.initialize_output_directory(seed_root)
        result_manager.save_dataset_properties_to_json(step_1_dir, initial_dataset_properties)

        cache_seed_dir = f"seed_{seed}"
        cached_best_weight_path = CACHE_ROOT / "baseline" / cache_seed_dir / "best.pt"
        cached_score_results_path = CACHE_ROOT / "scoring" / cache_seed_dir / "score_results.json"
        if not cached_best_weight_path.exists():
            raise FileNotFoundError(f"Missing cached baseline weight: {cached_best_weight_path}")
        if not cached_score_results_path.exists():
            raise FileNotFoundError(f"Missing cached scoring results: {cached_score_results_path}")

        baseline_model_config = copy.deepcopy(baseline_model_config_template)
        baseline_model_config.training_config.seed = int(seed)
        baseline_model = Baseline(baseline_model_config)

        best_weight_path = str(cached_best_weight_path)
        baseline_eval = baseline_model.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
        baseline_eval_dict = _to_eval_dict(baseline_eval)
        result_manager.save_evaluation_results_to_json(step_2_dir, baseline_eval, file_name="evaluation_baseline_on_original_test.json")

        score_results = _load_scoring_results(cached_score_results_path)
        result_manager.save_score_results_to_json(step_3_dir, score_results)

        summary["baseline"].append(
            {
                "seed": int(seed),
                "best_weight_path": str(best_weight_path),
                "baseline_eval_on_original_test": baseline_eval_dict,
                "selected_object_weight": float(score_results.selected_object_weight),
                "selected_false_positive_weight": float(score_results.selected_false_positive_weight),
                "scoring_weight_mode": str(score_results.scoring_weight_mode),
            }
        )

        for alpha in alphas:
            condition_name = _alpha_key(alpha)
            print(f"Seed {seed} - Running condition: {condition_name} (score_alpha={alpha})")

            augmentation_config = copy.deepcopy(augmentation_config_template)
            augmentation_config["use_score_guidance"] = True
            augmentation_config["selection_seed"] = int(seed)
            augmentation_config["score_alpha"] = float(alpha)

            condition_step_4_dir = step_4_dir / condition_name
            condition_step_5_dir = step_5_dir / condition_name
            condition_step_4_dir.mkdir(parents=True, exist_ok=True)
            condition_step_5_dir.mkdir(parents=True, exist_ok=True)

            augmentor = CopyPasteAugmentor(augmentation_config)
            augmented_dataset_dir = condition_step_4_dir / "augmented_dataset"
            new_dataset_properties = augmentor.create_new_dataset(
                initial_dataset_properties=initial_dataset_properties,
                scoring_results=score_results,
                new_dataset_path=augmented_dataset_dir,
            )
            result_manager.save_dataset_properties_to_json(condition_step_4_dir, new_dataset_properties)

            augmented_train_result_dir = condition_step_5_dir / "train_results"
            augmented_model = Baseline(baseline_model_config)
            best_weight_path_augmented = augmented_model.custom_train(new_dataset_properties, augmented_train_result_dir)
            augmented_eval = augmented_model.custom_evaluate_on_test_set(best_weight_path_augmented, initial_dataset_properties)
            augmented_eval_dict = _to_eval_dict(augmented_eval)

            result_manager.save_evaluation_results_to_json(
                condition_step_5_dir,
                augmented_eval,
                file_name="evaluation_augmented_on_original_test.json",
            )

            delta_eval = _delta_eval_dict(augmented_eval_dict, baseline_eval_dict)
            summary["conditions"][condition_name].append(
                {
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "use_score_guidance": True,
                    "best_weight_path_augmented": str(best_weight_path_augmented),
                    "baseline_eval_on_original_test": baseline_eval_dict,
                    "augmented_eval_on_original_test": augmented_eval_dict,
                    "delta_aug_minus_baseline": delta_eval,
                }
            )

    baseline_runs = summary["baseline"]
    summary["aggregate"]["baseline_eval_on_original_test"] = {
        "COCO_AP": _metric_stats([r["baseline_eval_on_original_test"]["COCO_AP"] for r in baseline_runs]),
        "COCO_AP50": _metric_stats([r["baseline_eval_on_original_test"]["COCO_AP50"] for r in baseline_runs]),
        "COCO_AP75": _metric_stats([r["baseline_eval_on_original_test"]["COCO_AP75"] for r in baseline_runs]),
        "AP_small": _metric_stats([r["baseline_eval_on_original_test"]["AP_small"] for r in baseline_runs]),
        "AP_medium": _metric_stats([r["baseline_eval_on_original_test"]["AP_medium"] for r in baseline_runs]),
        "AP_large": _metric_stats([r["baseline_eval_on_original_test"]["AP_large"] for r in baseline_runs]),
    }

    ranking = []
    for alpha in alphas:
        condition_name = _alpha_key(alpha)
        condition_runs = summary["conditions"][condition_name]
        condition_agg = _aggregate_condition(condition_runs)
        summary["aggregate"][condition_name] = condition_agg
        ranking.append(
            {
                "alpha": float(alpha),
                "condition": condition_name,
                "mean_delta_coco_ap": condition_agg["delta_aug_minus_baseline"]["COCO_AP"]["mean"],
                "std_delta_coco_ap": condition_agg["delta_aug_minus_baseline"]["COCO_AP"]["std"],
            }
        )

    summary["ranking_by_delta_coco_ap"] = sorted(
        ranking,
        key=lambda x: x["mean_delta_coco_ap"],
        reverse=True,
    )

    summary_path = RESULTS_DIR / "summary_alpha_sweep_3seeds.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFinished alpha sweep comparison. Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run copy-paste augmentation with score guidance for alpha sweep across 3 seeds "
            "while loading step 1-3 from cache."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR),
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Exactly 3 seeds as comma-separated values, e.g. 123,456,789. If omitted, first 3 seeds in general.random_seed are used.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="1,3,5",
        help="Comma-separated score_alpha values to test, e.g. 1,3,5.",
    )

    args = parser.parse_args()
    selected_seeds = _resolve_three_seeds(args.config, args.seeds)
    selected_alphas = _parse_alphas(args.alphas)
    run_experiment(args.config, selected_seeds, selected_alphas)