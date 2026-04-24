import argparse
import copy
import csv
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
DEFAULT_RATIOS = [0.5, 1.0, 2.0, 3.0]


def _metric_stats(values: list[float]) -> dict:
	if not values:
		return {"mean": 0.0, "std": 0.0}
	return {
		"mean": float(statistics.mean(values)),
		"std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
	}


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


def _ratio_tag(ratio: float) -> str:
	return f"ratio_{str(ratio).replace('.', 'p')}"


def _parse_seeds(config_path: str, override_seeds: str | None) -> list[int]:
	if override_seeds:
		seeds = [int(x.strip()) for x in override_seeds.split(",") if x.strip()]
		if not seeds:
			raise ValueError("--seeds cannot be empty")
		return seeds

	with open(config_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f) or {}
	random_seeds = (cfg.get("general") or {}).get("random_seed", [123])
	if isinstance(random_seeds, int):
		return [int(random_seeds)]
	if isinstance(random_seeds, list) and random_seeds:
		return [int(s) for s in random_seeds]
	return [123]


def _parse_ratios(override_ratios: str | None) -> list[float]:
	if not override_ratios:
		return list(DEFAULT_RATIOS)
	ratios = [float(x.strip()) for x in override_ratios.split(",") if x.strip()]
	if not ratios:
		raise ValueError("--ratios cannot be empty")
	if any(r <= 0 for r in ratios):
		raise ValueError("All dataset ratios must be > 0")
	return ratios


def run_experiment(config_path: str, seeds: list[int], ratios: list[float]):
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	exputils.copy_yaml_config(config_path, RESULTS_DIR / "frozen_config.yaml")

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

	print(f"Running ratio trend experiment with seeds={seeds} and ratios={ratios}")
	print(f"Auto max_det from p99 object count x3: {max_det}")

	summary: dict = {
		"seeds": [int(s) for s in seeds],
		"ratios": [float(r) for r in ratios],
		"baseline": [],
		"ratio_runs": [],
		"aggregate": {},
		"trend": [],
		"main_result": {},
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
		result_manager.save_evaluation_results_to_json(
			step_2_dir,
			baseline_eval,
			file_name="evaluation_baseline_on_original_test.json",
		)

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
			}
		)

		for ratio in ratios:
			ratio = float(ratio)
			ratio_name = _ratio_tag(ratio)
			print(f"Seed {seed} - Running dataset_ratio={ratio}")

			augmentation_config = copy.deepcopy(augmentation_config_template)
			augmentation_config["dataset_ratio"] = ratio
			augmentation_config["selection_seed"] = int(seed)

			ratio_step_4_dir = step_4_dir / ratio_name
			ratio_step_5_dir = step_5_dir / ratio_name
			ratio_step_4_dir.mkdir(parents=True, exist_ok=True)
			ratio_step_5_dir.mkdir(parents=True, exist_ok=True)

			augmentor = CopyPasteAugmentor(augmentation_config)
			augmented_dataset_dir = ratio_step_4_dir / "augmented_dataset"
			new_dataset_properties = augmentor.create_new_dataset(
				initial_dataset_properties=initial_dataset_properties,
				scoring_results=score_results,
				new_dataset_path=augmented_dataset_dir,
			)
			result_manager.save_dataset_properties_to_json(ratio_step_4_dir, new_dataset_properties)

			augmented_train_result_dir = ratio_step_5_dir / "train_results"
			augmented_model = Baseline(baseline_model_config)
			best_weight_path_augmented = augmented_model.custom_train(new_dataset_properties, augmented_train_result_dir)
			augmented_eval = augmented_model.custom_evaluate_on_test_set(best_weight_path_augmented, initial_dataset_properties)
			augmented_eval_dict = _to_eval_dict(augmented_eval)
			result_manager.save_evaluation_results_to_json(
				ratio_step_5_dir,
				augmented_eval,
				file_name="evaluation_augmented_on_original_test.json",
			)

			delta_eval = _delta_eval_dict(augmented_eval_dict, baseline_eval_dict)
			summary["ratio_runs"].append(
				{
					"seed": int(seed),
					"dataset_ratio": ratio,
					"ratio_name": ratio_name,
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

	for ratio in ratios:
		ratio = float(ratio)
		ratio_name = _ratio_tag(ratio)
		runs = [r for r in summary["ratio_runs"] if float(r["dataset_ratio"]) == ratio]
		summary["aggregate"][ratio_name] = {
			"dataset_ratio": ratio,
			"augmented_eval_on_original_test": {
				"COCO_AP": _metric_stats([r["augmented_eval_on_original_test"]["COCO_AP"] for r in runs]),
				"COCO_AP50": _metric_stats([r["augmented_eval_on_original_test"]["COCO_AP50"] for r in runs]),
				"COCO_AP75": _metric_stats([r["augmented_eval_on_original_test"]["COCO_AP75"] for r in runs]),
				"AP_small": _metric_stats([r["augmented_eval_on_original_test"]["AP_small"] for r in runs]),
				"AP_medium": _metric_stats([r["augmented_eval_on_original_test"]["AP_medium"] for r in runs]),
				"AP_large": _metric_stats([r["augmented_eval_on_original_test"]["AP_large"] for r in runs]),
			},
			"delta_aug_minus_baseline": {
				"COCO_AP": _metric_stats([r["delta_aug_minus_baseline"]["COCO_AP"] for r in runs]),
				"COCO_AP50": _metric_stats([r["delta_aug_minus_baseline"]["COCO_AP50"] for r in runs]),
				"COCO_AP75": _metric_stats([r["delta_aug_minus_baseline"]["COCO_AP75"] for r in runs]),
				"AP_small": _metric_stats([r["delta_aug_minus_baseline"]["AP_small"] for r in runs]),
				"AP_medium": _metric_stats([r["delta_aug_minus_baseline"]["AP_medium"] for r in runs]),
				"AP_large": _metric_stats([r["delta_aug_minus_baseline"]["AP_large"] for r in runs]),
			},
			"seeds_count": len(runs),
		}

		summary["trend"].append(
			{
				"dataset_ratio": ratio,
				"mean_delta_coco_ap": summary["aggregate"][ratio_name]["delta_aug_minus_baseline"]["COCO_AP"]["mean"],
				"std_delta_coco_ap": summary["aggregate"][ratio_name]["delta_aug_minus_baseline"]["COCO_AP"]["std"],
				"mean_augmented_coco_ap": summary["aggregate"][ratio_name]["augmented_eval_on_original_test"]["COCO_AP"]["mean"],
			}
		)

	sorted_trend = sorted(summary["trend"], key=lambda x: x["mean_delta_coco_ap"], reverse=True)
	best_ratio = sorted_trend[0]["dataset_ratio"] if sorted_trend else None
	summary["main_result"] = {
		"metric_for_claim": "delta_aug_minus_baseline.COCO_AP",
		"best_dataset_ratio_by_mean_delta": best_ratio,
		"trend_descending_by_mean_delta": sorted_trend,
		"seeds_count": len(seeds),
	}

	summary_path = RESULTS_DIR / "summary_ratio_sweep.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	csv_path = RESULTS_DIR / "trend_ratio_vs_delta.csv"
	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["dataset_ratio", "mean_delta_coco_ap", "std_delta_coco_ap", "mean_augmented_coco_ap"],
		)
		writer.writeheader()
		for row in sorted(summary["trend"], key=lambda x: x["dataset_ratio"]):
			writer.writerow(row)

	print(f"\nRatio sweep completed. Summary saved to: {summary_path}")
	print(f"Trend CSV saved to: {csv_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=(
			"Sweep copy-paste dataset_ratio values and evaluate trend of augmented-vs-baseline performance."
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
		help="Comma-separated seeds. If omitted, uses general.random_seed from config.",
	)
	parser.add_argument(
		"--ratios",
		type=str,
		default=None,
		help="Comma-separated dataset ratios, e.g. 0.5,1,2,3",
	)

	args = parser.parse_args()
	selected_seeds = _parse_seeds(args.config, args.seeds)
	selected_ratios = _parse_ratios(args.ratios)
	run_experiment(args.config, selected_seeds, selected_ratios)
