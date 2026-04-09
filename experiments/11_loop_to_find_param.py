import argparse
import copy
import itertools
import json
import random
from pathlib import Path
import sys
import traceback

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


RESULTS_DIR = Path(f"results/{Path(__file__).stem}")
CONFIG_DIR = Path("/home/khanh/Projects/DifficultyAgri/configs/experiments/minneapple_yolo.yaml")


def _load_single_seed(config_path: str | Path) -> int:
	with open(config_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f) or {}

	general_cfg = cfg.get("general") or {}
	seeds = general_cfg.get("random_seed")
	if isinstance(seeds, list) and len(seeds) > 0:
		return int(seeds[0])
	if isinstance(seeds, int):
		return int(seeds)
	return 123


def _serialize_eval(results_obj) -> dict:
	return {
		"COCO_AP": float(results_obj.COCO_AP),
		"COCO_AP50": float(results_obj.COCO_AP50),
		"COCO_AP75": float(results_obj.COCO_AP75),
		"AP_small": float(results_obj.AP_small),
		"AP_medium": float(results_obj.AP_medium),
		"AP_large": float(results_obj.AP_large),
	}


def _build_param_grid(mode: str) -> list[dict]:
	grid = {
		"dataset_ratio": [0.5, 1.0, 1.5, 2.0, 3.0],
		"target_density": [40, 50, 60, 70, 80],
		"max_paste_objects_per_image": [12, 16, 20, 24],
		"top_object_fraction": [0.5, 0.75, 1.0],
		"weight_scale": [1.5, 2.5, 3.5, 4.5],
		"background_weight_mode": ["linear", "exponential"],
		"object_weight_mode": ["linear", "exponential"],
		"max_background_reuse": [3, 4, 5, 6, None],
		"max_object_reuse": [3, 4, 5, 6, None],
	}

	keys = list(grid.keys())
	combinations = []
	for values in itertools.product(*(grid[k] for k in keys)):
		params = {k: v for k, v in zip(keys, values)}
		params["mode"] = mode
		combinations.append(params)
	return combinations


def _append_jsonl(path: Path, payload: dict) -> None:
	with open(path, "a", encoding="utf-8") as f:
		f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def _run_baseline_and_scoring(config_path: str):
	config_manager = ConfigManager()
	config_manager.load_all_configs(config_path)

	initial_dataset_config = config_manager.initial_dataset_config
	baseline_model_config = config_manager.baseline_config
	scoring_config = config_manager.scoring_config
	base_aug_config = dict(config_manager.augmentation_config or {})

	run_seed = _load_single_seed(config_path)
	baseline_model_config.training_config.seed = int(run_seed)

	result_manager = ResultManager()
	initial_dataset = CustomDataset(initial_dataset_config)
	initial_dataset.validate()
	initial_dataset_properties = initial_dataset.get_properties()

	baseline_dir = RESULTS_DIR / "baseline"
	baseline_dir.mkdir(parents=True, exist_ok=True)
	result_manager.save_dataset_properties_to_json(baseline_dir, initial_dataset_properties)

	train_result_dir = baseline_dir / "train_results"
	baseline_model = Baseline(baseline_model_config)
	best_weight_path = baseline_model.custom_train(initial_dataset_properties, train_result_dir)

	baseline_eval = baseline_model.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
	result_manager.save_evaluation_results_to_json(
		baseline_dir,
		baseline_eval,
		file_name="evaluation_report_baseline.json",
	)

	low_conf_threshold = 0.0001
	iou_threshold = 0.5
	max_det = compute_max_det_from_train_labels(
		train_labels_dir=initial_dataset_properties.train_labels_dir,
		percentile=0.99,
		multiplier=3,
	)

	image_dir = initial_dataset_properties.train_images_dir
	low_conf_prediction_dir = baseline_dir / "low_conf_predictions"
	low_conf_predictions = baseline_model.custom_predict(
		model_weight=best_weight_path,
		image_dir=image_dir,
		conf=low_conf_threshold,
		iou=iou_threshold,
		max_det=max_det,
	)
	result_manager.save_prediction_results(str(low_conf_prediction_dir), low_conf_predictions)

	optimal_conf_prediction_dir = baseline_dir / "optimal_conf_predictions"
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
	result_manager.save_prediction_results(str(optimal_conf_prediction_dir), optimal_conf_predictions)

	scoring = Scorer(scoring_config)
	score_results = scoring.score(
		str(optimal_conf_prediction_dir),
		str(low_conf_prediction_dir),
		images_dir=initial_dataset_properties.train_images_dir,
		labels_dir=initial_dataset_properties.train_labels_dir,
	)
	result_manager.save_score_results_to_json(str(baseline_dir), score_results, file_name="score_results.json")

	return {
		"dataset_properties": initial_dataset_properties,
		"baseline_model_config": baseline_model_config,
		"base_augmentation_config": base_aug_config,
		"score_results": score_results,
		"baseline_eval": baseline_eval,
		"baseline_best_weight": str(best_weight_path),
		"seed": int(run_seed),
	}


def _run_single_trial(
	trial_index: int,
	trial_params: dict,
	shared_ctx: dict,
) -> dict:
	trial_dir = RESULTS_DIR / "trials" / f"run_{trial_index:02d}"
	trial_dir.mkdir(parents=True, exist_ok=True)

	trial_aug_config = copy.deepcopy(shared_ctx["base_augmentation_config"])
	trial_aug_config.update(trial_params)

	with open(trial_dir / "trial_augmentation_config.json", "w", encoding="utf-8") as f:
		json.dump(trial_aug_config, f, indent=2)

	augmentor = CopyPasteAugmentor(trial_aug_config)
	augmented_dataset_dir = trial_dir / "augmented_dataset"
	new_dataset_properties = augmentor.create_new_dataset(
		initial_dataset_properties=shared_ctx["dataset_properties"],
		scoring_results=shared_ctx["score_results"],
		new_dataset_path=str(augmented_dataset_dir),
	)

	model_config = copy.deepcopy(shared_ctx["baseline_model_config"])
	model_config.training_config.seed = int(shared_ctx["seed"])
	model = Baseline(model_config)

	train_dir = trial_dir / "train_results"
	best_weight_path = model.custom_train(new_dataset_properties, train_dir)

	eval_test = model.custom_evaluate_on_test_set(best_weight_path, shared_ctx["dataset_properties"])

	result_manager = ResultManager()
	result_manager.save_dataset_properties_to_json(str(trial_dir), new_dataset_properties)
	result_manager.save_evaluation_results_to_json(
		str(trial_dir),
		eval_test,
		file_name="evaluation_report_test_set.json",
	)

	result = {
		"run_index": int(trial_index),
		"params": trial_params,
		"best_weight_path": str(best_weight_path),
		"eval_on_test_set": _serialize_eval(eval_test),
		"trial_dir": str(trial_dir),
	}

	with open(trial_dir / "trial_result.json", "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2)

	return result


def run_search(
	config_path: str,
	max_runs: int,
	mode: str,
	metric: str,
	seed: int,
	baseline_override: float | None = None,
) -> None:
	if max_runs < 0:
		raise ValueError("max_runs must be >= 0")

	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	search_config = {
		"config_path": str(config_path),
		"max_runs": int(max_runs),
		"mode": mode,
		"metric": metric,
		"search_seed": int(seed),
		"baseline_override": baseline_override,
	}
	_write_json(RESULTS_DIR / "search_config.json", search_config)

	shared_ctx = _run_baseline_and_scoring(config_path)
	baseline_eval_dict = _serialize_eval(shared_ctx["baseline_eval"])
	if metric not in baseline_eval_dict:
		raise ValueError(f"Unknown metric '{metric}'. Available: {list(baseline_eval_dict.keys())}")

	baseline_metric = float(baseline_eval_dict[metric])
	if baseline_override is not None:
		baseline_metric = float(baseline_override)

	all_candidates = _build_param_grid(mode=mode)
	rng = random.Random(seed)
	rng.shuffle(all_candidates)
	total_candidates = len(all_candidates)
	run_limit = total_candidates if max_runs == 0 else min(int(max_runs), total_candidates)

	log_path = RESULTS_DIR / "run_log.jsonl"
	best_realtime_path = RESULTS_DIR / "best_realtime.json"
	checkpoint_path = RESULTS_DIR / "search_summary_checkpoint.json"
	if log_path.exists():
		log_path.unlink()

	print("=" * 80)
	print("Parameter search started")
	print(f"Metric: {metric}")
	print(f"Baseline {metric}: {baseline_metric:.6f}")
	print(f"Total combinations: {total_candidates}")
	print(f"Runs to execute: {run_limit}")
	if max_runs == 0:
		print("Mode: run all combinations")
	print("=" * 80)

	all_results: list[dict] = []
	failed_results: list[dict] = []
	best_result: dict | None = None

	_append_jsonl(
		log_path,
		{
			"event": "search_started",
			"metric": metric,
			"baseline_metric": baseline_metric,
			"total_combinations": total_candidates,
			"run_limit": run_limit,
			"mode": mode,
		},
	)

	for idx, params in enumerate(all_candidates[:run_limit], start=1):

		print("\n" + "-" * 80)
		print(f"Run {idx}/{run_limit}")
		print(json.dumps(params, indent=2))
		_append_jsonl(
			log_path,
			{
				"event": "run_started",
				"run_index": idx,
				"total_runs": run_limit,
				"params": params,
			},
		)

		try:
			trial_result = _run_single_trial(
				trial_index=idx,
				trial_params=params,
				shared_ctx=shared_ctx,
			)
			trial_metric = float(trial_result["eval_on_test_set"][metric])
			improvement = trial_metric - baseline_metric

			trial_result["metric_name"] = metric
			trial_result["metric_value"] = trial_metric
			trial_result["baseline_metric"] = baseline_metric
			trial_result["improvement_over_baseline"] = improvement

			all_results.append(trial_result)
			if best_result is None or trial_metric > float(best_result["metric_value"]):
				best_result = trial_result
				_write_json(
					best_realtime_path,
					{
						"metric": metric,
						"baseline_metric": baseline_metric,
						"best_result": best_result,
						"runs_finished": len(all_results) + len(failed_results),
						"run_limit": run_limit,
					},
				)

			_append_jsonl(
				log_path,
				{
					"event": "run_completed",
					"run_index": idx,
					"metric": metric,
					"metric_value": trial_metric,
					"improvement_over_baseline": improvement,
					"is_best_so_far": best_result is not None and best_result["run_index"] == idx,
				},
			)

			print(
				f"Run {idx}: {metric}={trial_metric:.6f} | "
				f"baseline={baseline_metric:.6f} | delta={improvement:+.6f}"
			)
		except Exception as exc:
			failure = {
				"run_index": idx,
				"params": params,
				"error": str(exc),
				"traceback": traceback.format_exc(),
			}
			failed_results.append(failure)
			_append_jsonl(
				log_path,
				{
					"event": "run_failed",
					"run_index": idx,
					"params": params,
					"error": str(exc),
				},
			)
			print(f"Run {idx} failed: {exc}")

		checkpoint = {
			"metric": metric,
			"baseline_metric": baseline_metric,
			"baseline_eval": baseline_eval_dict,
			"runs_finished": len(all_results) + len(failed_results),
			"runs_succeeded": len(all_results),
			"runs_failed": len(failed_results),
			"run_limit": run_limit,
			"best_result": best_result,
			"last_success_result": all_results[-1] if all_results else None,
			"last_failure": failed_results[-1] if failed_results else None,
		}
		_write_json(checkpoint_path, checkpoint)

	search_summary = {
		"metric": metric,
		"baseline_metric": baseline_metric,
		"baseline_eval": baseline_eval_dict,
		"runs_executed": len(all_results) + len(failed_results),
		"runs_succeeded": len(all_results),
		"runs_failed": len(failed_results),
		"run_limit": int(run_limit),
		"total_combinations": int(total_candidates),
		"best_result": best_result,
		"failed_results": failed_results,
		"all_results": all_results,
	}

	_write_json(RESULTS_DIR / "search_summary.json", search_summary)
	_append_jsonl(
		log_path,
		{
			"event": "search_completed",
			"runs_executed": len(all_results) + len(failed_results),
			"runs_succeeded": len(all_results),
			"runs_failed": len(failed_results),
			"best_run_index": None if best_result is None else best_result["run_index"],
		},
	)

	print("\n" + "=" * 80)
	print("Search completed")
	print(f"Executed runs: {len(all_results) + len(failed_results)}/{run_limit}")
	print(f"Succeeded: {len(all_results)}, Failed: {len(failed_results)}")
	if best_result is not None:
		print(
			f"Best run: {best_result['run_index']} | "
			f"{metric}={best_result['metric_value']:.6f} | "
			f"delta={best_result['improvement_over_baseline']:+.6f}"
		)
	print(f"Summary saved to: {RESULTS_DIR / 'search_summary.json'}")
	print(f"Realtime best saved to: {best_realtime_path}")
	print(f"Crash-safe log saved to: {log_path}")
	print("=" * 80)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description=(
			"Loop over augmentation parameter combinations and report the best result. "
			"Use --max-runs 0 to run all combinations."
		)
	)
	parser.add_argument(
		"--config",
		type=str,
		default=str(CONFIG_DIR),
		help="Path to configuration YAML file",
	)
	parser.add_argument(
		"--max-runs",
		type=int,
		default=20,
		help="Maximum number of trials; set 0 to run all parameter combinations",
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="difficulty_based_copy_paste",
		help="Augmentation mode to search",
	)
	parser.add_argument(
		"--metric",
		type=str,
		default="COCO_AP",
		help="Metric to optimize (e.g. COCO_AP, COCO_AP50, AP_small)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=123,
		help="Random seed for shuffling combination order",
	)
	parser.add_argument(
		"--baseline-ap",
		type=float,
		default=None,
		help="Optional override baseline metric value for comparison",
	)
	args = parser.parse_args()

	run_search(
		config_path=args.config,
		max_runs=args.max_runs,
		mode=args.mode,
		metric=args.metric,
		seed=args.seed,
		baseline_override=args.baseline_ap,
	)
