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
from dagri.data import CustomDataset, compute_max_det_from_train_labels
from dagri.baseline import Baseline
from dagri.scoring.scorer import Scorer
from dagri.weighting import scoring_results_to_score_map, scores_to_weight_map, weight_summary

import experiments.utils as exputils


RESULTS_DIR = Path(f"results/{Path(__file__).stem}")
CONFIG_DIR = Path("/home/khanh/Projects/DifficultyAgri/configs/experiments/global_wheat_head_yolo.yaml")


def _to_eval_dict(evaluation_results) -> dict:
    return {
        "COCO_AP": float(evaluation_results.COCO_AP),
        "COCO_AP50": float(evaluation_results.COCO_AP50),
        "COCO_AP75": float(evaluation_results.COCO_AP75),
        "AP_small": float(evaluation_results.AP_small),
        "AP_medium": float(evaluation_results.AP_medium),
        "AP_large": float(evaluation_results.AP_large),
    }


def _metric_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean_val = float(statistics.mean(values))
    std_val = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean_val, "std": std_val}


def run_experiment(config_path: str):
    step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, logs_dir = exputils.initialize_output_directory(RESULTS_DIR)
    frozen_config_path = Path(RESULTS_DIR) / "frozen_config.yaml"
    exputils.copy_yaml_config(config_path, frozen_config_path)

    config_manager = ConfigManager()
    config_manager.load_all_configs(config_path)
    initial_dataset_config = config_manager.initial_dataset_config
    baseline_model_config = config_manager.baseline_config
    scoring_config = config_manager.scoring_config
    augmentation_config = config_manager.augmentation_config or {}

    result_manager = ResultManager()

    initial_dataset = CustomDataset(initial_dataset_config)
    initial_dataset.validate()
    initial_dataset_properties = initial_dataset.get_properties()
    result_manager.save_dataset_properties_to_json(step_1_dir, initial_dataset_properties)

    low_conf_thershold = 0.0001
    iou_threshold = 0.5
    max_det = compute_max_det_from_train_labels(
        train_labels_dir=initial_dataset_properties.train_labels_dir,
        percentile=0.99,
        multiplier=3,
    )
    print(f"Auto max_det from p99 object count x3: {max_det}")

    baseline_model = Baseline(baseline_model_config)
    baseline_train_dir = step_2_dir / "train_results"
    baseline_best_path = baseline_train_dir / "best.pt"
    if baseline_best_path.exists():
        print(f"Reusing existing baseline best weights: {baseline_best_path}")
        best_weight_path = str(baseline_best_path)
    else:
        best_weight_path = baseline_model.custom_train(initial_dataset_properties, baseline_train_dir)
    baseline_eval = baseline_model.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
    result_manager.save_evaluation_results_to_json(step_2_dir, baseline_eval, file_name="evaluation_baseline_on_test.json")

    score_results_path = step_3_dir / "score_results.json"
    if score_results_path.exists():
        print(f"Reusing existing scoring results: {score_results_path}")
        with open(score_results_path, "r", encoding="utf-8") as f:
            score_results = json.load(f)
        optimal_conf_threshold = -1.0
    else:
        image_dir = initial_dataset_properties.train_images_dir
        low_conf_prediction_dir = f"{step_2_dir}/low_conf_predictions"
        low_conf_predictions = baseline_model.custom_predict(
            model_weight=best_weight_path,
            image_dir=image_dir,
            conf=low_conf_thershold,
            iou=iou_threshold,
            max_det=max_det,
        )
        result_manager.save_prediction_results(low_conf_prediction_dir, low_conf_predictions)

        optimal_conf_threshold_prediction_dir = f"{step_2_dir}/optimal_conf_predictions"
        optimal_conf_threshold = baseline_model.get_optimal_conf_threshold_for_scoring(
            dataset_properties=initial_dataset_properties,
            model_weight=best_weight_path,
        )
        optimal_conf_predictions = baseline_model.custom_predict(
            model_weight=best_weight_path,
            image_dir=image_dir,
            conf=optimal_conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
        )
        result_manager.save_prediction_results(optimal_conf_threshold_prediction_dir, optimal_conf_predictions)

        scoring = Scorer(scoring_config)
        score_results = scoring.score(
            optimal_conf_threshold_prediction_dir,
            low_conf_prediction_dir,
            images_dir=initial_dataset_properties.train_images_dir,
            labels_dir=initial_dataset_properties.train_labels_dir,
        )
        result_manager.save_score_results_to_json(step_3_dir, score_results)

    image_score_map = scoring_results_to_score_map(score_results)
    image_weight_map = scores_to_weight_map(
        image_score_map,
        function_name=str(augmentation_config.get("score_weight_function", "linear")),
        gamma=float(augmentation_config.get("score_alpha", 1.0)),
        normalize=True,
    )
    with open(step_3_dir / "image_weight_map.json", "w", encoding="utf-8") as f:
        json.dump(image_weight_map, f, indent=2)
    with open(step_3_dir / "weight_summary.json", "w", encoding="utf-8") as f:
        json.dump(weight_summary(image_weight_map), f, indent=2)

    weighted_model_config = copy.deepcopy(baseline_model_config)
    weighted_model_config.training_config.weighted_sampling = True
    weighted_model_config.training_config.weighted_sampling_function = str(augmentation_config.get("score_weight_function", "linear"))
    weighted_model_config.training_config.weighted_sampling_gamma = float(augmentation_config.get("score_alpha", 1.0))
    weighted_model_config.training_config.weighted_sampling_normalize = True

    weighted_model = Baseline(weighted_model_config)
    weighted_train_dir = step_4_dir / "train_results"
    weighted_best_weight_path = weighted_model.custom_train(
        initial_dataset_properties,
        weighted_train_dir,
        image_score_map=image_score_map,
        weight_function=weighted_model_config.training_config.weighted_sampling_function,
        weight_gamma=weighted_model_config.training_config.weighted_sampling_gamma,
        normalize_scores=weighted_model_config.training_config.weighted_sampling_normalize,
    )
    weighted_eval = weighted_model.custom_evaluate_on_test_set(weighted_best_weight_path, initial_dataset_properties)
    result_manager.save_evaluation_results_to_json(step_4_dir, weighted_eval, file_name="evaluation_weighted_on_test.json")

    summary = {
        "baseline": {
            "best_weight_path": str(best_weight_path),
            "evaluation": _to_eval_dict(baseline_eval),
            "optimal_conf_threshold": float(optimal_conf_threshold),
        },
        "weighted": {
            "best_weight_path": str(weighted_best_weight_path),
            "evaluation": _to_eval_dict(weighted_eval),
            "weight_function": weighted_model_config.training_config.weighted_sampling_function,
            "weight_gamma": float(weighted_model_config.training_config.weighted_sampling_gamma),
            "weight_summary": weight_summary(image_weight_map),
        },
        "delta_weighted_minus_baseline": {
            key: float(_to_eval_dict(weighted_eval)[key] - _to_eval_dict(baseline_eval)[key])
            for key in _to_eval_dict(baseline_eval)
        },
        "max_det": int(max_det),
    }
    summary["aggregate"] = {
        "baseline": {
            key: _metric_stats([value]) for key, value in _to_eval_dict(baseline_eval).items()
        },
        "weighted": {
            key: _metric_stats([value]) for key, value in _to_eval_dict(weighted_eval).items()
        },
    }

    summary_path = RESULTS_DIR / "summary_weighted_training.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Weighted training experiment completed. Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run weighted YOLO training from scored images")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR),
        help="Path to the experiment config YAML file.",
    )
    args = parser.parse_args()
    run_experiment(args.config)
