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
from dagri.data import CustomDataset, compute_max_det_from_train_labels
from dagri.baseline import Baseline
from dagri.scoring.scorer import Scorer
from dagri.augmentation import CopyPasteAugmentor
from dagri.data.dota_utils import (
	download_dota_dataset,
	convert_dota_to_yolo,
	get_dota_class_names
)

import experiments.utils as exputils


RESULTS_DIR = Path(f"results/{Path(__file__).stem}")
CONFIG_DIR = Path("/home/khanh/Projects/DifficultyAgri/configs/experiments/dota_yolo.yaml")
DATASETS_DIR = Path("/home/khanh/Projects/DifficultyAgri/datasets/dota")


def _load_seeds(config_path: str | Path) -> list[int]:
	with open(config_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f) or {}

	general_cfg = cfg.get("general") or {}
	raw = general_cfg.get("random_seed", [123, 456, 789])
	if isinstance(raw, list) and raw:
		return [int(x) for x in raw[:3]]
	if isinstance(raw, int):
		return [int(raw)]
	return [123, 456, 789]


def _serialize_eval(results_obj) -> dict:
	return {
		"COCO_AP": float(results_obj.COCO_AP),
		"COCO_AP50": float(results_obj.COCO_AP50),
		"COCO_AP75": float(results_obj.COCO_AP75),
		"AP_small": float(results_obj.AP_small),
		"AP_medium": float(results_obj.AP_medium),
		"AP_large": float(results_obj.AP_large),
	}


def prepare_dota_dataset(
	dota_version: str = "v2",
	download_dir: str | None = None,
	output_dir: str | None = None
) -> str:
	"""
	Prepare DOTA dataset: download, convert to YOLO format.

	Args:
		dota_version: "v1" or "v2"
		download_dir: Directory where DOTA is downloaded (if None, assumes already downloaded)
		output_dir: Directory to save YOLO-formatted dataset

	Returns:
		Path to YOLO-formatted dataset
	"""
	if output_dir is None:
		output_dir = str(DATASETS_DIR / "yolo_format")

	output_path = Path(output_dir) / f"dota_yolo"
	print(f"\n{'='*60}")
	print(f"DOTA Dataset Preparation")
	print(f"{'='*60}")

	# Check if already converted
	if (output_path / "train" / "images").exists() and \
	   (output_path / "train" / "labels").exists() and \
	   (output_path / "val" / "images").exists():
		print(f"✓ DOTA dataset already in YOLO format at {output_path}")
		return str(output_path)

	# Download if needed
	if download_dir is None:
		download_dir = str(DATASETS_DIR / "raw")

	dota_dir = download_dota_dataset(download_dir, version=dota_version)

	# Check if raw dataset exists
	dota_path = Path(dota_dir)
	if not dota_path.exists():
		print(f"\n❌ DOTA dataset not found at {dota_path}")
		print(f"\nPlease download DOTA-{dota_version} manually:")
		print(f"1. Visit https://captain-whu.github.io/DOTA/")
		print(f"2. Download DOTA-{dota_version} dataset")
		print(f"3. Extract to {download_dir}")
		print(f"\nExpected structure:")
		print(f"  {dota_path}/")
		print(f"  ├── train/")
		print(f"  │   ├── images/")
		print(f"  │   └── labelTxt/")
		print(f"  ├── val/")
		print(f"  │   ├── images/")
		print(f"  │   └── labelTxt/")
		print(f"  └── test/")
		print(f"      └── images/")
		raise FileNotFoundError(f"DOTA dataset not found. Please download it manually.")

	# Convert to YOLO format
	print(f"\nConverting DOTA to YOLO format...")
	convert_dota_to_yolo(str(dota_path), str(output_path), version=dota_version)

	return str(output_path)


def run_experiment(config_path: str, dota_version: str = "v2") -> None:
	"""
	Run full DOTA pipeline: prepare dataset -> train -> score -> augment -> retrain.

	Args:
		config_path: Path to configuration YAML file
		dota_version: DOTA version to use ("v1" or "v2")
	"""
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	frozen_config_path = RESULTS_DIR / "frozen_config.yaml"
	exputils.copy_yaml_config(config_path, frozen_config_path)

	# Prepare dataset
	print("\nStep 0: Prepare DOTA Dataset")
	print("-" * 60)
	dataset_path = prepare_dota_dataset(dota_version=dota_version)

	# Update config with actual dataset path
	with open(config_path, "r", encoding="utf-8") as f:
		config_data = yaml.safe_load(f)
	config_data["dataset_config"]["root_dir"] = dataset_path
	with open(frozen_config_path, "w", encoding="utf-8") as f:
		yaml.safe_dump(config_data, f)

	seeds = _load_seeds(config_path)
	print(f"\n✓ Dataset ready. Running full pipeline with seeds: {seeds}")

	# Load static config once.
	config_manager = ConfigManager()
	config_manager.load_all_configs(str(frozen_config_path))
	initial_dataset_config = config_manager.initial_dataset_config
	baseline_model_config_template = config_manager.baseline_config
	scoring_config = config_manager.scoring_config
	augmentation_config = config_manager.augmentation_config

	result_manager = ResultManager()

	# Dataset is seed-independent; validate once and reuse properties.
	print("\nStep 1: Load and Validate Dataset")
	print("-" * 60)
	initial_dataset = CustomDataset(initial_dataset_config)
	initial_dataset.validate()
	initial_dataset_properties = initial_dataset.get_properties()
	print(f"✓ Dataset validated")
	print(f"  Train: {len(list(Path(initial_dataset_properties.train_images_dir).glob('*')))} images")
	print(f"  Val: {len(list(Path(initial_dataset_properties.val_images_dir).glob('*')))} images")
	print(f"  Test: {len(list(Path(initial_dataset_properties.test_images_dir).glob('*')))} images")

	aggregate = {
		"seeds": seeds,
		"runs": [],
		"mean": {},
	}

	for seed in seeds:
		print(f"\n{'='*60}")
		print(f"Processing Seed: {seed}")
		print(f"{'='*60}")

		seed_root = RESULTS_DIR / f"seed_{seed}"
		step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, logs_dir = \
			exputils.initialize_output_directory(seed_root)

		# Save dataset properties under each seed folder for reproducibility.
		result_manager.save_dataset_properties_to_json(step_1_dir, initial_dataset_properties)

		baseline_model_config = copy.deepcopy(baseline_model_config_template)
		baseline_model_config.training_config.seed = int(seed)

		# Step 2: baseline train + evaluate.
		print(f"\nStep 2: Train and Evaluate Baseline Model")
		print("-" * 60)
		train_result_dir = step_2_dir / "train_results"
		baseline_model = Baseline(baseline_model_config)
		best_weight_path = baseline_model.custom_train(initial_dataset_properties, train_result_dir)
		print(f"✓ Baseline training complete")

		evaluation_results_baseline = baseline_model.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
		result_manager.save_evaluation_results_to_json(step_2_dir, evaluation_results_baseline)
		print(f"✓ Baseline evaluation complete (AP: {evaluation_results_baseline.COCO_AP:.4f})")

		# Step 3: scoring inputs + scoring.
		print(f"\nStep 3: Score Dataset for Difficult Samples")
		print("-" * 60)
		low_conf_threshold = 0.0001
		iou_threshold = 0.5
		max_det = compute_max_det_from_train_labels(
			train_labels_dir=initial_dataset_properties.train_labels_dir,
			percentile=0.99,
			multiplier=3,
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
		print(f"✓ Scoring complete")

		# Step 4: copy-paste augmentation.
		print(f"\nStep 4: Copy-Paste Augmentation")
		print("-" * 60)
		augmentor = CopyPasteAugmentor(augmentation_config)
		augmented_dataset_dir = step_4_dir / "augmented_dataset"
		new_dataset_properties = augmentor.create_new_dataset(
			initial_dataset_properties=initial_dataset_properties,
			scoring_results=score_results,
			new_dataset_path=augmented_dataset_dir,
		)
		result_manager.save_dataset_properties_to_json(step_4_dir, new_dataset_properties)
		print(f"✓ Augmentation complete")

		# Step 5: retrain on augmented dataset + compare on original test set.
		print(f"\nStep 5: Train and Evaluate Model on Augmented Dataset")
		print("-" * 60)
		new_train_result_dir = step_5_dir / "train_results"
		baseline_model_aug = Baseline(baseline_model_config)
		best_weight_path_new_dataset = baseline_model_aug.custom_train(new_dataset_properties, new_train_result_dir)
		print(f"✓ Augmented training complete")

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
		print(f"✓ Evaluation complete")
		print(f"  Baseline AP: {evaluation_results_baseline.COCO_AP:.4f}")
		print(f"  Augmented AP: {evaluation_results_new_dataset.COCO_AP:.4f}")
		print(f"  Improvement: {(evaluation_results_new_dataset.COCO_AP - evaluation_results_baseline.COCO_AP):.4f}")

		aggregate["runs"].append(
			{
				"seed": int(seed),
				"optimal_conf_threshold": float(optimal_conf_threshold),
				"baseline_eval": _serialize_eval(evaluation_results_baseline),
				"recheck_initial_eval": _serialize_eval(evaluation_results_initial_dataset),
				"augmented_eval": _serialize_eval(evaluation_results_new_dataset),
				"selected_object_weight": float(score_results.selected_object_weight),
				"selected_false_positive_weight": float(score_results.selected_false_positive_weight),
				"scoring_weight_mode": str(score_results.scoring_weight_mode),
			}
		)

	# Aggregate mean metrics for quick comparison.
	if aggregate["runs"]:
		keys = ["COCO_AP", "COCO_AP50", "COCO_AP75", "AP_small", "AP_medium", "AP_large"]
		mean_baseline = {}
		mean_aug = {}
		for key in keys:
			mean_baseline[key] = float(sum(r["baseline_eval"][key] for r in aggregate["runs"]) / len(aggregate["runs"]))
			mean_aug[key] = float(sum(r["augmented_eval"][key] for r in aggregate["runs"]) / len(aggregate["runs"]))
		aggregate["mean"] = {
			"baseline_eval": mean_baseline,
			"augmented_eval": mean_aug,
		}

	summary_path = RESULTS_DIR / "summary_3_seed.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(aggregate, f, indent=2)

	print(f"\n{'='*60}")
	print(f"✓ DOTA full experiment completed!")
	print(f"{'='*60}")
	print(f"Summary saved to: {summary_path}")
	print(f"\nMean Results (3 seeds):")
	print(f"  Baseline AP: {aggregate['mean']['baseline_eval']['COCO_AP']:.4f}")
	print(f"  Augmented AP: {aggregate['mean']['augmented_eval']['COCO_AP']:.4f}")
	print(f"  Improvement: {(aggregate['mean']['augmented_eval']['COCO_AP'] - aggregate['mean']['baseline_eval']['COCO_AP']):.4f}")
	print(f"  Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Run full DOTA pipeline (3 seeds): prepare dataset -> baseline -> scoring -> copy-paste -> retrain"
	)
	parser.add_argument(
		"--config",
		type=str,
		default=str(CONFIG_DIR),
		help="Path to configuration YAML file",
	)
	parser.add_argument(
		"--dota-version",
		type=str,
		default="v2",
		choices=["v1", "v2"],
		help="DOTA dataset version to use",
	)
	args = parser.parse_args()
	run_experiment(args.config, dota_version=args.dota_version)
