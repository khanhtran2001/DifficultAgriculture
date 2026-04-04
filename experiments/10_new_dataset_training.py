import argparse
import copy
import json
from pathlib import Path
import sys
import yaml

# Allow running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from dagri.general.config_manager import ConfigManager
from dagri.general.result_manager import ResultManager
from dagri.data import CustomDataset
from dagri.baseline import Baseline
from dagri.interfaces import DatasetProperties

import experiments.utils as exputils


RESULTS_DIR = Path(f"results/{Path(__file__).stem}")
CONFIG_DIR = Path("/home/khanh/Projects/DifficultyAgri/configs/experiments/minneapple_yolo.yaml")
DEFAULT_AUGMENTED_ROOT = Path(
	"/home/khanh/Projects/DifficultyAgri/results/09_aug_exp_only/seed_123/Step_4_Copy_Paste_Augmentation/augmented_dataset"
)


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


def _resolve_augmented_train_dirs(augmented_root: Path) -> tuple[Path, Path]:
	"""
	Resolve train image/label directories from either:
	- <root>/train/images and <root>/train/labels
	- <root>/images and <root>/labels
	"""
	candidates = [
		(augmented_root / "train" / "images", augmented_root / "train" / "labels"),
		(augmented_root / "images", augmented_root / "labels"),
	]

	for img_dir, lbl_dir in candidates:
		if img_dir.exists() and lbl_dir.exists():
			return img_dir, lbl_dir

	raise FileNotFoundError(
		"Could not find augmented train folders. Expected either "
		f"'{augmented_root / 'train' / 'images'}' + '{augmented_root / 'train' / 'labels'}' "
		"or '<root>/images' + '<root>/labels'."
	)


def _serialize_eval(results_obj) -> dict:
	return {
		"COCO_AP": float(results_obj.COCO_AP),
		"COCO_AP50": float(results_obj.COCO_AP50),
		"COCO_AP75": float(results_obj.COCO_AP75),
		"AP_small": float(results_obj.AP_small),
		"AP_medium": float(results_obj.AP_medium),
		"AP_large": float(results_obj.AP_large),
	}


def run_experiment(config_path: str, augmented_train_root: str) -> None:
	# Initialize output directory tree and freeze config.
	step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, logs_dir = exputils.initialize_output_directory(RESULTS_DIR)
	frozen_config_path = Path(RESULTS_DIR) / "frozen_config.yaml"
	exputils.copy_yaml_config(config_path, frozen_config_path)

	# Load configuration blocks.
	config_manager = ConfigManager()
	config_manager.load_all_configs(config_path)
	initial_dataset_config = config_manager.initial_dataset_config
	baseline_model_config = config_manager.baseline_config

	# Single-seed run for reproducibility.
	run_seed = _load_single_seed(config_path)
	baseline_model_config.training_config.seed = int(run_seed)
	print(f"Running training-only pipeline with seed: {run_seed}")

	result_manager = ResultManager()

	# Step 1: load and validate original dataset properties once (for val/test paths).
	initial_dataset = CustomDataset(initial_dataset_config)
	initial_dataset.validate()
	initial_dataset_properties = initial_dataset.get_properties()
	result_manager.save_dataset_properties_to_json(step_1_dir, initial_dataset_properties)

	# Build new dataset properties using cached augmented TRAIN split.
	augmented_root = Path(augmented_train_root).resolve()
	aug_train_images_dir, aug_train_labels_dir = _resolve_augmented_train_dirs(augmented_root)

	new_dataset_properties = copy.deepcopy(initial_dataset_properties)
	new_dataset_properties.root_dir = str(augmented_root)
	new_dataset_properties.train_images_dir = str(aug_train_images_dir)
	new_dataset_properties.train_labels_dir = str(aug_train_labels_dir)

	print("=" * 70)
	print("Training with cached augmented dataset")
	print("=" * 70)
	print(f"Augmented root: {augmented_root}")
	print(f"Train images:   {new_dataset_properties.train_images_dir}")
	print(f"Train labels:   {new_dataset_properties.train_labels_dir}")

	result_manager.save_dataset_properties_to_json(step_5_dir, new_dataset_properties)

	# Step 5 equivalent: train baseline using new dataset only.
	train_result_dir = step_5_dir / "train_results"
	baseline_model_aug = Baseline(baseline_model_config)
	best_weight_path_new_dataset = baseline_model_aug.custom_train(new_dataset_properties, train_result_dir)

	eval_new = baseline_model_aug.custom_evaluate_on_test_set(best_weight_path_new_dataset, new_dataset_properties)
	result_manager.save_evaluation_results_to_json(
		step_5_dir,
		eval_new,
		file_name="evaluation_report_on_new_dataset.json",
	)

	# Save quick summary.
	summary = {
		"seed": int(run_seed),
		"augmented_train_root": str(augmented_root),
		"train_images_dir": str(aug_train_images_dir),
		"train_labels_dir": str(aug_train_labels_dir),
		"best_weight_path": str(best_weight_path_new_dataset),
		"eval_on_new_dataset": _serialize_eval(eval_new),
	}

	summary_path = RESULTS_DIR / "training_only_summary.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print("\n" + "=" * 70)
	print("✓ Training-only experiment completed successfully")
	print(f"Summary saved to: {summary_path}")
	print("=" * 70)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Train baseline model using cached augmented train split only"
	)
	parser.add_argument(
		"--config",
		type=str,
		default=str(CONFIG_DIR),
		help="Path to configuration YAML file",
	)
	parser.add_argument(
		"--augmented-train-root",
		type=str,
		default=str(DEFAULT_AUGMENTED_ROOT),
		help="Root directory containing augmented train data",
	)
	args = parser.parse_args()

	run_experiment(args.config, args.augmented_train_root)
