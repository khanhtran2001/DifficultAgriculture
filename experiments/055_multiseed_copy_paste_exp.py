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
from dagri.augmentation import CopyPasteAugmentor

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
        raise ValueError(
            "Please provide at least one random seed in 'general.random_seed' "
            "as an int or a non-empty list of ints."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frozen_config_path = Path(RESULTS_DIR) / "frozen_config.yaml"
    exputils.copy_yaml_config(config_path, frozen_config_path)

    config_manager = ConfigManager()
    config_manager.load_all_configs(config_path)
    initial_dataset_config = config_manager.initial_dataset_config
    baseline_model_config_template = config_manager.baseline_config
    scoring_config = config_manager.scoring_config
    augmentation_config = config_manager.augmentation_config

    result_manager = ResultManager()

    initial_dataset = CustomDataset(initial_dataset_config)
    initial_dataset.validate()
    initial_dataset_properties = initial_dataset.get_properties()
    low_conf_thershold = 0.0001
    iou_threshold = 0.5
    max_det = compute_max_det_from_train_labels(
        train_labels_dir=initial_dataset_properties.train_labels_dir,
        percentile=0.99,
        multiplier=3,
    )
    image_dir = initial_dataset_properties.train_images_dir

    print(f"Running multi-seed copy-paste experiment with seeds: {seeds}")
    print(f"Auto max_det from p99 object count x3: {max_det}")
    summary = {
        "seeds": seeds,
        "runs": [],
        "aggregate": {},
        "main_result": {},
    }

    for seed in seeds:
        print(f"\n========== Running seed {seed} ==========")
        seed_root = RESULTS_DIR / f"seed_{seed}"
        step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, logs_dir = exputils.initialize_output_directory(seed_root)
        result_manager.save_dataset_properties_to_json(step_1_dir, initial_dataset_properties)

        baseline_model_config = copy.deepcopy(baseline_model_config_template)
        baseline_model_config.training_config.seed = int(seed)
        baseline_model = Baseline(baseline_model_config)

        train_result_dir = step_2_dir / "train_results"
        best_weight_path = baseline_model.custom_train(initial_dataset_properties, train_result_dir)
        baseline_eval = baseline_model.custom_evaluate_on_test_set(best_weight_path, initial_dataset_properties)
        result_manager.save_evaluation_results_to_json(step_2_dir, baseline_eval)

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

        augmentor = CopyPasteAugmentor(augmentation_config)
        augmented_dataset_dir = step_4_dir / "augmented_dataset"
        new_dataset_properties = augmentor.create_new_dataset(
            initial_dataset_properties=initial_dataset_properties,
            scoring_results=score_results,
            new_dataset_path=augmented_dataset_dir,
        )
        result_manager.save_dataset_properties_to_json(step_4_dir, new_dataset_properties)

        augmented_train_result_dir = step_5_dir / "train_results"
        augmented_model = Baseline(baseline_model_config)
        best_weight_path_augmented = augmented_model.custom_train(new_dataset_properties, augmented_train_result_dir)

        # Evaluate both models on the same original test split for direct comparison.
        augmented_eval_on_original_test = augmented_model.custom_evaluate_on_test_set(
            best_weight_path_augmented,
            initial_dataset_properties,
        )
        result_manager.save_evaluation_results_to_json(
            step_5_dir,
            augmented_eval_on_original_test,
            file_name="evaluation_augmented_model_on_original_test.json",
        )

        delta_coco_ap = float(augmented_eval_on_original_test.COCO_AP - baseline_eval.COCO_AP)
        delta_ap50 = float(augmented_eval_on_original_test.COCO_AP50 - baseline_eval.COCO_AP50)
        delta_ap75 = float(augmented_eval_on_original_test.COCO_AP75 - baseline_eval.COCO_AP75)
        delta_ap_small = float(augmented_eval_on_original_test.AP_small - baseline_eval.AP_small)
        delta_ap_medium = float(augmented_eval_on_original_test.AP_medium - baseline_eval.AP_medium)
        delta_ap_large = float(augmented_eval_on_original_test.AP_large - baseline_eval.AP_large)

        summary["runs"].append(
            {
                "seed": int(seed),
                "best_weight_path_baseline": str(best_weight_path),
                "best_weight_path_augmented": str(best_weight_path_augmented),
                "optimal_conf_threshold": float(optimal_conf_threshold),
                "selected_object_weight": float(score_results.selected_object_weight),
                "selected_false_positive_weight": float(score_results.selected_false_positive_weight),
                "scoring_weight_mode": str(score_results.scoring_weight_mode),
                "baseline_eval_on_original_test": {
                    "COCO_AP": float(baseline_eval.COCO_AP),
                    "COCO_AP50": float(baseline_eval.COCO_AP50),
                    "COCO_AP75": float(baseline_eval.COCO_AP75),
                    "AP_small": float(baseline_eval.AP_small),
                    "AP_medium": float(baseline_eval.AP_medium),
                    "AP_large": float(baseline_eval.AP_large),
                },
                "augmented_eval_on_original_test": {
                    "COCO_AP": float(augmented_eval_on_original_test.COCO_AP),
                    "COCO_AP50": float(augmented_eval_on_original_test.COCO_AP50),
                    "COCO_AP75": float(augmented_eval_on_original_test.COCO_AP75),
                    "AP_small": float(augmented_eval_on_original_test.AP_small),
                    "AP_medium": float(augmented_eval_on_original_test.AP_medium),
                    "AP_large": float(augmented_eval_on_original_test.AP_large),
                },
                "delta_aug_minus_baseline": {
                    "COCO_AP": delta_coco_ap,
                    "COCO_AP50": delta_ap50,
                    "COCO_AP75": delta_ap75,
                    "AP_small": delta_ap_small,
                    "AP_medium": delta_ap_medium,
                    "AP_large": delta_ap_large,
                },
            }
        )

    if summary["runs"]:
        summary["aggregate"] = {
            "baseline_eval_on_original_test": {
                "COCO_AP": _metric_stats([r["baseline_eval_on_original_test"]["COCO_AP"] for r in summary["runs"]]),
                "COCO_AP50": _metric_stats([r["baseline_eval_on_original_test"]["COCO_AP50"] for r in summary["runs"]]),
                "COCO_AP75": _metric_stats([r["baseline_eval_on_original_test"]["COCO_AP75"] for r in summary["runs"]]),
                "AP_small": _metric_stats([r["baseline_eval_on_original_test"]["AP_small"] for r in summary["runs"]]),
                "AP_medium": _metric_stats([r["baseline_eval_on_original_test"]["AP_medium"] for r in summary["runs"]]),
                "AP_large": _metric_stats([r["baseline_eval_on_original_test"]["AP_large"] for r in summary["runs"]]),
            },
            "augmented_eval_on_original_test": {
                "COCO_AP": _metric_stats([r["augmented_eval_on_original_test"]["COCO_AP"] for r in summary["runs"]]),
                "COCO_AP50": _metric_stats([r["augmented_eval_on_original_test"]["COCO_AP50"] for r in summary["runs"]]),
                "COCO_AP75": _metric_stats([r["augmented_eval_on_original_test"]["COCO_AP75"] for r in summary["runs"]]),
                "AP_small": _metric_stats([r["augmented_eval_on_original_test"]["AP_small"] for r in summary["runs"]]),
                "AP_medium": _metric_stats([r["augmented_eval_on_original_test"]["AP_medium"] for r in summary["runs"]]),
                "AP_large": _metric_stats([r["augmented_eval_on_original_test"]["AP_large"] for r in summary["runs"]]),
            },
            "delta_aug_minus_baseline": {
                "COCO_AP": _metric_stats([r["delta_aug_minus_baseline"]["COCO_AP"] for r in summary["runs"]]),
                "COCO_AP50": _metric_stats([r["delta_aug_minus_baseline"]["COCO_AP50"] for r in summary["runs"]]),
                "COCO_AP75": _metric_stats([r["delta_aug_minus_baseline"]["COCO_AP75"] for r in summary["runs"]]),
                "AP_small": _metric_stats([r["delta_aug_minus_baseline"]["AP_small"] for r in summary["runs"]]),
                "AP_medium": _metric_stats([r["delta_aug_minus_baseline"]["AP_medium"] for r in summary["runs"]]),
                "AP_large": _metric_stats([r["delta_aug_minus_baseline"]["AP_large"] for r in summary["runs"]]),
            },
        }

        summary["main_result"] = {
            "metric_for_claim": "delta_aug_minus_baseline.COCO_AP",
            "mean_delta_coco_ap": summary["aggregate"]["delta_aug_minus_baseline"]["COCO_AP"]["mean"],
            "std_delta_coco_ap": summary["aggregate"]["delta_aug_minus_baseline"]["COCO_AP"]["std"],
            "seeds_count": len(summary["runs"]),
        }

    summary_path = RESULTS_DIR / "summary_multiseed_copy_paste.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMulti-seed copy-paste experiment completed. Summary saved to: {summary_path}")

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