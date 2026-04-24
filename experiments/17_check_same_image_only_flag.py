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
from dagri.augmentation import CopyPasteAugmentor

import experiments.utils as exputils


RESULTS_DIR = Path(f"results/{Path(__file__).stem}")
CONFIG_DIR = Path("/home/khanh/Projects/DifficultyAgri/configs/experiments/global_wheat_head_yolo.yaml")
DEFAULT_CONDITIONS = {
	"same_image_only_on": True,
	"same_image_only_off": False,
}


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


def run_experiment(config_path: str, seeds: list[int]):
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	frozen_config_path = RESULTS_DIR / "frozen_config.yaml"
	exputils.copy_yaml_config(config_path, frozen_config_path)

	config_manager = ConfigManager()
	config_manager.load_all_configs(config_path)

	initial_dataset_config = config_manager.initial_dataset_config
	baseline_model_config_template = config_manager.baseline_config
	scoring_config = config_manager.scoring_config
	augmentation_config_template = dict(config_manager.augmentation_config or {})

	result_manager = ResultManager()

	initial_dataset = CustomDataset(initial_dataset_config)
	initial_dataset.validate()
	initial_dataset_properties = initial_dataset.get_properties()

	low_conf_threshold = 0.0001
	iou_threshold = 0.5
	max_det = compute_max_det_from_train_labels(
		train_labels_dir=initial_dataset_properties.train_labels_dir,
		percentile=0.99,
		multiplier=3,
	)

	print(f"Running same_image_only comparison with seeds: {seeds}")
	print(f"Auto max_det from p99 object count x3: {max_det}")

	summary: dict = {
		"hypothesis": {
			"h0": "Turning same_image_only on does not improve COCO_AP delta vs baseline.",
			"h1": "Turning same_image_only on improves COCO_AP delta vs baseline.",
			"metric_for_claim": "delta_aug_minus_baseline.COCO_AP",
		},
		"seeds": seeds,
		"conditions": {name: [] for name in DEFAULT_CONDITIONS},
		"baseline": [],
		"aggregate": {},
		"comparison": {},
	}

	for seed in seeds:
		print(f"\n========== Seed {seed} ==========")
		seed_root = RESULTS_DIR / f"seed_{seed}"
		step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, _ = exputils.initialize_output_directory(seed_root)
		result_manager.save_dataset_properties_to_json(step_1_dir, initial_dataset_properties)

		baseline_model_config = copy.deepcopy(baseline_model_config_template)
		baseline_model_config.training_config.seed = int(seed)
		baseline_model = Baseline(baseline_model_config)

		train_result_dir = step_2_dir / "train_results"
		best_weight_path = baseline_model.custom_train(initial_dataset_properties, train_result_dir)
		baseline_eval = baseline_model.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
		baseline_eval_dict = _to_eval_dict(baseline_eval)
		result_manager.save_evaluation_results_to_json(step_2_dir, baseline_eval, file_name="evaluation_baseline_on_original_test.json")

		image_dir = initial_dataset_properties.train_images_dir
		low_conf_prediction_dir = f"{step_2_dir}/low_conf_predictions"
		low_conf_predictions = baseline_model.custom_predict(
			model_weight=best_weight_path,
			image_dir=image_dir,
			conf=low_conf_threshold,
			iou=iou_threshold,
			max_det=max_det,
		)
		result_manager.save_prediction_results(low_conf_prediction_dir, low_conf_predictions)

		optimal_conf_prediction_dir = f"{step_2_dir}/optimal_conf_predictions"
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
		result_manager.save_prediction_results(optimal_conf_prediction_dir, optimal_conf_predictions)
		print(f"Seed {seed} - Optimal confidence threshold: {optimal_conf_threshold}")

		scoring = Scorer(scoring_config)
		score_results = scoring.score(
			optimal_conf_prediction_dir,
			low_conf_prediction_dir,
			images_dir=initial_dataset_properties.train_images_dir,
			labels_dir=initial_dataset_properties.train_labels_dir,
		)
		result_manager.save_score_results_to_json(step_3_dir, score_results)

		summary["baseline"].append(
			{
				"seed": int(seed),
				"best_weight_path": str(best_weight_path),
				"optimal_conf_threshold": float(optimal_conf_threshold),
				"baseline_eval_on_original_test": baseline_eval_dict,
				"selected_object_weight": float(score_results.selected_object_weight),
				"selected_false_positive_weight": float(score_results.selected_false_positive_weight),
				"scoring_weight_mode": str(score_results.scoring_weight_mode),
			}
		)

		for condition_name, same_image_only in DEFAULT_CONDITIONS.items():
			print(f"Seed {seed} - Running condition: {condition_name} (same_image_only={same_image_only})")

			augmentation_config = copy.deepcopy(augmentation_config_template)
			augmentation_config["same_image_only"] = bool(same_image_only)
			augmentation_config["selection_seed"] = int(seed)

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
					"same_image_only": bool(same_image_only),
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

	for condition_name in DEFAULT_CONDITIONS:
		condition_runs = summary["conditions"][condition_name]
		summary["aggregate"][condition_name] = _aggregate_condition(condition_runs)

	on_delta = summary["aggregate"]["same_image_only_on"]["delta_aug_minus_baseline"]
	off_delta = summary["aggregate"]["same_image_only_off"]["delta_aug_minus_baseline"]
	summary["comparison"] = {
		"metric_for_claim": "delta_aug_minus_baseline.COCO_AP",
		"same_image_only_on_mean_delta_coco_ap": on_delta["COCO_AP"]["mean"],
		"same_image_only_off_mean_delta_coco_ap": off_delta["COCO_AP"]["mean"],
		"mean_delta_advantage_same_image_only_on": (
			on_delta["COCO_AP"]["mean"] - off_delta["COCO_AP"]["mean"]
		),
		"same_image_only_on_std_delta_coco_ap": on_delta["COCO_AP"]["std"],
		"same_image_only_off_std_delta_coco_ap": off_delta["COCO_AP"]["std"],
		"seeds_count": len(seeds),
		"recommendation": (
			"same_image_only_on"
			if on_delta["COCO_AP"]["mean"] >= off_delta["COCO_AP"]["mean"]
			else "same_image_only_off"
		),
	}

	summary_path = RESULTS_DIR / "summary_same_image_only_on_vs_off_3seeds.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print(f"\nFinished same_image_only flag comparison. Summary saved to: {summary_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=(
			"Run copy-paste augmentation and evaluation for 3 seeds with two conditions: "
			"same_image_only on and same_image_only off, then compare means."
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

	args = parser.parse_args()
	selected_seeds = _resolve_three_seeds(args.config, args.seeds)
	run_experiment(args.config, selected_seeds)
