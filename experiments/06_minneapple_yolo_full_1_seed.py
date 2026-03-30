import argparse
from pathlib import Path
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


def run_experiment(config_path: str) -> None:
	# Initialize output directory tree and freeze config.
	step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, logs_dir = exputils.initialize_output_directory(RESULTS_DIR)
	frozen_config_path = Path(RESULTS_DIR) / "frozen_config.yaml"
	exputils.copy_yaml_config(config_path, frozen_config_path)

	# Load configuration blocks.
	config_manager = ConfigManager()
	config_manager.load_all_configs(config_path)
	initial_dataset_config = config_manager.initial_dataset_config
	baseline_model_config = config_manager.baseline_config
	scoring_config = config_manager.scoring_config
	augmentation_config = config_manager.augmentation_config

	# Force single-seed run from general.random_seed[0] if present.
	run_seed = _load_single_seed(config_path)
	baseline_model_config.training_config.seed = int(run_seed)
	print(f"Running full pipeline with single seed: {run_seed}")

	result_manager = ResultManager()

	# Step 1: dataset validate and properties.
	initial_dataset = CustomDataset(initial_dataset_config)
	initial_dataset.validate()
	initial_dataset_properties = initial_dataset.get_properties()
	result_manager.save_dataset_properties_to_json(step_1_dir, initial_dataset_properties)

	# Step 2: baseline training + evaluation.
	train_result_dir = step_2_dir / "train_results"
	baseline_model = Baseline(baseline_model_config)
	best_weight_path = baseline_model.custom_train(initial_dataset_properties, train_result_dir)

	evaluation_results_baseline = baseline_model.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
	result_manager.save_evaluation_results_to_json(step_2_dir, evaluation_results_baseline)

	# Step 3: generate train predictions at low conf and optimal conf, then score dataset.
	low_conf_threshold = 0.0001
	iou_threshold = 0.5
	max_det = compute_max_det_from_train_labels(
		train_labels_dir=initial_dataset_properties.train_labels_dir,
		percentile=0.99,
		multiplier=3,
	)
	print(f"Auto max_det from p99 object count x3: {max_det}")

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
	print(f"Optimal confidence threshold: {optimal_conf_threshold}")

	scoring = Scorer(scoring_config)
	score_results = scoring.score(
		optimal_conf_threshold_prediction_dir,
		low_conf_prediction_dir,
		images_dir=initial_dataset_properties.train_images_dir,
		labels_dir=initial_dataset_properties.train_labels_dir,
	)
	result_manager.save_score_results_to_json(step_3_dir, score_results)

	# Step 4: copy-paste augmentation.
	augmentor = CopyPasteAugmentor(augmentation_config)
	augmented_dataset_dir = step_4_dir / "augmented_dataset"
	new_dataset_properties = augmentor.create_new_dataset(
		initial_dataset_properties=initial_dataset_properties,
		scoring_results=score_results,
		new_dataset_path=augmented_dataset_dir,
	)
	result_manager.save_dataset_properties_to_json(step_4_dir, new_dataset_properties)

	# Step 5: retrain on augmented dataset and compare evaluations.
	new_train_result_dir = step_5_dir / "train_results"
	baseline_model_aug = Baseline(baseline_model_config)
	best_weight_path_new_dataset = baseline_model_aug.custom_train(new_dataset_properties, new_train_result_dir)

	evaluation_results_initial_dataset = baseline_model_aug.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
	result_manager.save_evaluation_results_to_json(
		step_5_dir,
		evaluation_results_initial_dataset,
		file_name="evaluation_report_initial_dataset.json",
	)

	evaluation_results_new_dataset = baseline_model_aug.custom_evaluate_on_test_set(best_weight_path_new_dataset, new_dataset_properties)
	result_manager.save_evaluation_results_to_json(
		step_5_dir,
		evaluation_results_new_dataset,
		file_name="evaluation_report_new_dataset.json",
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run full MinneApple pipeline (single-seed): baseline -> scoring -> copy-paste -> retrain")
	parser.add_argument(
		"--config",
		type=str,
		default=str(CONFIG_DIR),
		help="Path to configuration YAML file",
	)
	args = parser.parse_args()
	run_experiment(args.config)

