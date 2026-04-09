import argparse
import json
from pathlib import Path
import sys

# Allow running this file directly: `python experiments/01_minneapple_yolo_augmentation.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dagri.general.config_manager import ConfigManager
from dagri.general.result_manager import ResultManager
from dagri.data import compute_max_det_from_train_labels
from dagri.baseline import Baseline
from dagri.interfaces import DatasetProperties
from dagri.scoring.scorer import Scorer

import experiments.utils as exputils
"""
Output directory structure:
results/02_only_scoring/
    ├── frozen_config.yaml
    ├── Step_1_Load_and_Validate_Dataset/
    ├── Step_2_Train_and_Evaluate_BASELINE_MODEL/
    ├── Step_3_Scoring_Dataset/
    └── logs/

This experiment reuses dataset metadata and trained weights from a previous
experiment run and does not retrain the model.
"""

# The parent result dir is in the folder results/exp_name

RESULTS_DIR = Path(f"results/{Path(__file__).stem}")
CONFIG_DIR = Path("/home/khanh/Projects/DifficultyAgri/configs/experiments/global_wheat_head_yolo.yaml")
DATASET_PROPERTIES_PATH = Path(
    "/home/khanh/Projects/DifficultyAgri/results/01_only_training/seed_123/Step_1_Load_and_Validate_Dataset/dataset_properties.json"
)
MODEL_WEIGHT_PATH = Path(
    "/home/khanh/Projects/DifficultyAgri/results/01_only_training/seed_123/Step_2_Train_and_Evaluate_BASELINE_MODEL/train_results/best.pt"
)


def load_dataset_properties(dataset_properties_file: str | Path) -> DatasetProperties:
    """Load dataset properties from a JSON file."""
    with open(dataset_properties_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetProperties(**data)


def resolve_input_paths(dataset_properties_path: str | Path, model_weight_path: str | Path) -> tuple[Path, Path]:
    """Validate and normalize dataset properties and trained model weight paths."""
    dataset_properties_file = Path(dataset_properties_path)
    best_weight_path = Path(model_weight_path)

    if not dataset_properties_file.exists():
        raise FileNotFoundError(f"Dataset properties file not found: {dataset_properties_file}")

    if not best_weight_path.exists():
        raise FileNotFoundError(f"Trained model weights not found: {best_weight_path}")

    return dataset_properties_file, best_weight_path



def run_experiment(config_path: str, dataset_properties_path: str, model_weight_path: str):

    # Initialize ouptut directory
    step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, logs_dir = exputils.initialize_output_directory(RESULTS_DIR) 
    # Frozen the configuration file for reproducibility
    frozen_config_path = Path(RESULTS_DIR) / "frozen_config.yaml"
    exputils.copy_yaml_config(config_path, frozen_config_path)

    # Load configuration
    config_manager = ConfigManager()
    config_manager.load_all_configs(config_path)
    baseline_model_config = config_manager.baseline_config
    scoring_config = config_manager.scoring_config

    # Initialize result manager
    result_manager = ResultManager()

    # Load dataset properties and trained weights from the previous experiment run.
    dataset_properties_file, best_weight_path = resolve_input_paths(dataset_properties_path, model_weight_path)
    initial_dataset_properties = load_dataset_properties(dataset_properties_file)

    # Save the loaded dataset properties into this experiment's output for traceability.
    result_manager.save_dataset_properties_to_json(step_1_dir, initial_dataset_properties)

    baseline_model = Baseline(baseline_model_config)

    # Evaluate the reused model on the test set.
    evaluation_results = baseline_model.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
    result_manager.save_evaluation_results_to_json(step_2_dir, evaluation_results)

    # Score the dataset using the trained baseline model and save results
    
    LOW_CONF_THRESHOLD = 0.0001
    IOU_THRESHOLD = 0.5 # Non-maximum suppression IoU threshold

    # First we need to get the predictions of the baseline model on the train set to use as reference for scoring
    max_det = compute_max_det_from_train_labels(
        train_labels_dir=initial_dataset_properties.train_labels_dir,
        percentile=0.99,
        multiplier=3,
    )
    print(f"Auto max_det from p99 object count x3: {max_det}")
    image_dir = initial_dataset_properties.train_images_dir
    low_conf_prediction_dir = f"{step_2_dir}/low_conf_predictions"
    low_conf_predictions = baseline_model.custom_predict(model_weight=best_weight_path, image_dir=image_dir, conf=LOW_CONF_THRESHOLD, iou=IOU_THRESHOLD, max_det=max_det)
    result_manager.save_prediction_results(low_conf_prediction_dir, low_conf_predictions)

    # Next we need to find the optimal confidence threshold that gives us the best score for the dataset
    optimal_conf_threshold_prediction_dir = f"{step_2_dir}/optimal_conf_predictions"
    optimal_conf_threshold = baseline_model.get_optimal_conf_threshold_for_scoring(dataset_properties=initial_dataset_properties, model_weight=best_weight_path) # Find the optimal confidence threshold that balances precision and recall for the dataset, this will be use the validation set to find the optimal confidence threshold.
    optimal_conf_predictions = baseline_model.custom_predict(model_weight=best_weight_path, image_dir=image_dir, conf=optimal_conf_threshold, iou=IOU_THRESHOLD, max_det=max_det)
    result_manager.save_prediction_results(optimal_conf_threshold_prediction_dir, optimal_conf_predictions)
    print(f"Optimal confidence threshold: {optimal_conf_threshold}")

    # Then we can score the dataset using the predictions and save the results
    """
    This score will do the following:
    1. Compare the predictions with the ground truth labels to calculate true positives, false positives.
    """
    scoring = Scorer(scoring_config)

    score_results = scoring.score(
        optimal_conf_threshold_prediction_dir,
        low_conf_prediction_dir,
        images_dir=initial_dataset_properties.train_images_dir,
        labels_dir=initial_dataset_properties.train_labels_dir,
    )
    result_manager.save_score_results_to_json(step_3_dir, score_results)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MinneApple YOLO augmentation experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR),
        help="Path to the configuration YAML file (default: hardcoded CONFIG_DIR)"
    )
    parser.add_argument(
        "--dataset-properties",
        type=str,
        default=str(DATASET_PROPERTIES_PATH),
        help="Path to dataset_properties.json from a previous training run",
    )
    parser.add_argument(
        "--model-weight",
        type=str,
        default=str(MODEL_WEIGHT_PATH),
        help="Path to trained best.pt from a previous training run",
    )
    args = parser.parse_args()
    run_experiment(args.config, args.dataset_properties, args.model_weight)