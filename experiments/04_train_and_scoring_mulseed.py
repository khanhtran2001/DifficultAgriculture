import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import yaml

# Allow running this file directly: `python experiments/01_minneapple_yolo_augmentation.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dagri.general.config_manager import ConfigManager
from dagri.general.result_manager import ResultManager
from dagri.data import CustomDataset, compute_max_det_from_train_labels
from dagri.baseline import Baseline
from dagri.scoring.scorer import Scorer

import experiments.utils as exputils
"""
Ouutput directory structure:
outputs/
    01_minneapple_yolo_augmentation/
    ├── frozen_config.yaml
    ├── Step_1_Load_and_Validate_Dataset/
        ├── dataset_config.yaml
        ├── validation_report.json
    ├── Step_2_Train_and_Evaluate_Baseline_Model/
        ├── model_config.yaml
        ├── best_model.pt
    ├── Step_3_Scoring_Dataset/
        ├── scoring_config.yaml
        ├── detail_score.json
    ├── Step_4_Copy_Paste_Augmentation/
        ├── augmentation_config.yaml
        ├── augmented_dataset/
    ├── Step_5_Train_and_Evaluate_Model_on_Augmented_Dataset/
        ├── model_config.yaml
        ├── augmented_best_model.pt
        ├── evaluation_report.json
    ├── logs/
        ├── dataset_log.txt
        ├── Initial_Training_Log.txt
        ├── Scoring_Log.txt
        ├── Augmentation_Log.txt
        ├── Evaluation_Log.txt
"""

# The parent result dir is in the folder results/exp_name

RESULTS_DIR = Path(f"results/{Path(__file__).stem}")
CONFIG_DIR = Path("/home/khanh/Projects/DifficultyAgri/configs/experiments/minneapple_yolo.yaml")


def _metric_stats(values: list[float]) -> dict:
    """Return mean/std summary for a metric list."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean_val = float(statistics.mean(values))
    std_val = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean_val, "std": std_val}



def run_experiment(config_path: str):

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    general_cfg = cfg.get("general") or {}
    random_seeds = general_cfg.get("random_seed", [123])
    if isinstance(random_seeds, int):
        seeds = [int(random_seeds)]
    elif isinstance(random_seeds, list) and random_seeds:
        seeds = [int(s) for s in random_seeds]
    else:
        raise ValueError("Please provide at least one random seed in the configuration under 'general.random_seed' as an int or a non-empty list of ints.")

    # Initialize output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Frozen the configuration file for reproducibility
    frozen_config_path = Path(RESULTS_DIR) / "frozen_config.yaml"
    exputils.copy_yaml_config(config_path, frozen_config_path)

    # Load configuration
    config_manager = ConfigManager()
    config_manager.load_all_configs(config_path)
    initial_dataset_config = config_manager.initial_dataset_config
    baseline_model_config_template = config_manager.baseline_config
    scoring_config = config_manager.scoring_config

    # Initialize result manager
    result_manager = ResultManager()

    # Initialize dataset
    initial_dataset = CustomDataset(initial_dataset_config)
    # Validate dataset and save results
    initial_dataset.validate()
    initial_dataset_properties = initial_dataset.get_properties()
    low_conf_thershold = 0.0001
    iou_threshold = 0.5
    max_det = compute_max_det_from_train_labels(
        train_labels_dir=initial_dataset_properties.train_labels_dir,
        percentile=0.99,
        multiplier=3,
    )
    print(f"Running Step 1-3 with seeds: {seeds}")
    print(f"Auto max_det from p99 object count x3: {max_det}")

    summary = {
        "seeds": seeds,
        "runs": [],
        "aggregate": {},
    }

    for seed in seeds:
        seed_root = RESULTS_DIR / f"seed_{seed}"
        step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, logs_dir = exputils.initialize_output_directory(seed_root)

        # Step 1: Save dataset properties per seed for reproducibility.
        result_manager.save_dataset_properties_to_json(step_1_dir, initial_dataset_properties)

        # Step 2: Train baseline and evaluate on test set.
        baseline_model_config = copy.deepcopy(baseline_model_config_template)
        baseline_model_config.training_config.seed = int(seed)
        train_result_dir = step_2_dir / "train_results"
        baseline_model = Baseline(baseline_model_config)
        best_weight_path = baseline_model.custom_train(initial_dataset_properties, train_result_dir)

        evaluation_results = baseline_model.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
        result_manager.save_evaluation_results_to_json(step_2_dir, evaluation_results)

        # Step 3: Predict and score on train split.
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
        print(f"Seed {seed} - Optimal confidence threshold: {optimal_conf_threshold}")

        scoring = Scorer(scoring_config)
        score_results = scoring.score(
            optimal_conf_threshold_prediction_dir,
            low_conf_prediction_dir,
            images_dir=initial_dataset_properties.train_images_dir,
            labels_dir=initial_dataset_properties.train_labels_dir,
        )
        result_manager.save_score_results_to_json(step_3_dir, score_results)

        summary["runs"].append(
            {
                "seed": int(seed),
                "best_weight_path": str(best_weight_path),
                "optimal_conf_threshold": float(optimal_conf_threshold),
                "evaluation": {
                    "COCO_AP": float(evaluation_results.COCO_AP),
                    "COCO_AP50": float(evaluation_results.COCO_AP50),
                    "COCO_AP75": float(evaluation_results.COCO_AP75),
                    "AP_small": float(evaluation_results.AP_small),
                    "AP_medium": float(evaluation_results.AP_medium),
                    "AP_large": float(evaluation_results.AP_large),
                },
                "selected_object_weight": float(score_results.selected_object_weight),
                "selected_false_positive_weight": float(score_results.selected_false_positive_weight),
                "scoring_weight_mode": str(score_results.scoring_weight_mode),
            }
        )

    if summary["runs"]:
        summary["aggregate"] = {
            "evaluation": {
                "COCO_AP": _metric_stats([r["evaluation"]["COCO_AP"] for r in summary["runs"]]),
                "COCO_AP50": _metric_stats([r["evaluation"]["COCO_AP50"] for r in summary["runs"]]),
                "COCO_AP75": _metric_stats([r["evaluation"]["COCO_AP75"] for r in summary["runs"]]),
                "AP_small": _metric_stats([r["evaluation"]["AP_small"] for r in summary["runs"]]),
                "AP_medium": _metric_stats([r["evaluation"]["AP_medium"] for r in summary["runs"]]),
                "AP_large": _metric_stats([r["evaluation"]["AP_large"] for r in summary["runs"]]),
            },
            "scoring": {
                "optimal_conf_threshold": _metric_stats([r["optimal_conf_threshold"] for r in summary["runs"]]),
                "selected_object_weight": _metric_stats([r["selected_object_weight"] for r in summary["runs"]]),
                "selected_false_positive_weight": _metric_stats([
                    r["selected_false_positive_weight"] for r in summary["runs"]
                ]),
            },
        }

    summary_path = RESULTS_DIR / "summary_multiseed_scoring.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Multi-seed Step 1-3 experiment completed. Summary saved to: {summary_path}")

    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MinneApple YOLO augmentation experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR),
        help="Path to the configuration YAML file (default: hardcoded CONFIG_DIR)"
    )
    args = parser.parse_args()
    run_experiment(args.config)