import yaml
import json
import shutil
from pathlib import Path

def initialize_output_directory(parent_output_dir: Path, overwrite: bool = True) -> tuple[Path, Path, Path, Path, Path, Path]:
    """
    Initialize the output directory for the experiment.
    If overwrite is True and directory exists, delete and recreate it.
    """
    if parent_output_dir.exists():
        if overwrite:
            shutil.rmtree(parent_output_dir)
            parent_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory {parent_output_dir} existed and was overwritten.")
        else:
            print(f"Output directory {parent_output_dir} already exists. Reusing it.")
    else:
        parent_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"New output directory {parent_output_dir} has been created.")


    step_1_dir = parent_output_dir / "Step_1_Load_and_Validate_Dataset"
    step_2_dir = parent_output_dir / "Step_2_Train_and_Evaluate_BASELINE_MODEL"
    step_3_dir = parent_output_dir / "Step_3_Scoring_Dataset"
    step_4_dir = parent_output_dir / "Step_4_Copy_Paste_Augmentation"
    step_5_dir = parent_output_dir / "Step_5_Train_and_Evaluate_Model_on_Augmented_Dataset"
    logs_dir = parent_output_dir / "logs"
    step_1_dir.mkdir(parents=True, exist_ok=True)
    step_2_dir.mkdir(parents=True, exist_ok=True)
    step_3_dir.mkdir(parents=True, exist_ok=True)
    step_4_dir.mkdir(parents=True, exist_ok=True)
    step_5_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Subdirectories for the experiment have been created.")
    return step_1_dir, step_2_dir, step_3_dir, step_4_dir, step_5_dir, logs_dir

def copy_yaml_config(config_path: Path, output_path: Path) -> None:
    """
    Copy a YAML configuration file to a new location.
    """
    shutil.copy(config_path, output_path)
    print(f"Configuration file {config_path} has been copied to {output_path}.") 

def load_yaml_config(config_path: Path) -> dict:
    """
    Load a YAML configuration file and return it as a dictionary.
    """
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    general_config = config.get("general_config")
    dataset_config = config.get("dataset_config")
    baseline_model_config = config.get("baseline_model_config")
    scoring_config = config.get("scoring_config")
    augmentation_config = config.get("augmentation_config")

    return general_config, dataset_config, baseline_model_config, scoring_config, augmentation_config
