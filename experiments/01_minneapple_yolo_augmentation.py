import argparse
from pathlib import Path
import sys
import os

# Allow running this file directly: `python experiments/01_minneapple_yolo_augmentation.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dagri.general.config_manage import ConfigManager
from dagri.general.result_manager import ResultManager
from dagri.data import CustomDataset
from dagri.baseline import Baseline

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

RESULTS_DIR = Path("results/01_minneapple_yolo_copy_paste_augmentation_detection")



def run_experiment(config_path: str):

    
    # Initialize ouptut directory
    step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, logs_dir = exputils.initialize_output_directory(RESULTS_DIR) 
    # Frozen the configuration file for reproducibility
    frozen_config_path = Path(RESULTS_DIR) / "frozen_config.yaml"
    exputils.copy_yaml_config(config_path, frozen_config_path)

    # Load configuration
    config_manager = ConfigManager()
    config_manager.load_all_configs(config_path)
    initial_dataset_config = config_manager.initial_dataset_config
    baseline_model_config = config_manager.baseline_config

    # Initialize result manager
    result_manager = ResultManager()

    # Initialize dataset
    initial_dataset = CustomDataset(initial_dataset_config)
    initial_dataset.validate()
    # Save the dataset properties to the output directory
    initial_dataset_properties = initial_dataset.get_properties()
    result_manager.save_dataset_properties_to_json(initial_dataset_properties, step_1_dir)

    # Train and evaluate baseline model using dataset properties
    baseline_model = Baseline(baseline_model_config, initial_dataset_properties)
    best_weight_path = baseline_model.custom_train(step_2_dir)

    # Evaluate model on test set from best trained weights
    baseline_model.custom_evaluate_on_test_set(best_weight_path, step_2_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    run_experiment(args.config)