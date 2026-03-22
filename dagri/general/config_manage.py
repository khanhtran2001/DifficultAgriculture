import os
from pathlib import Path
import json
import yaml

from dagri.interfaces import DatasetConfig, BaselineConfig

class ConfigManager:
    def __init__(self):
        self.initial_dataset_config = None
        self.baseline_config = None
    
    def load_all_configs(self, yaml_config_file_path: str) -> None:
        with open(yaml_config_file_path, "r") as f:
            all_configs = yaml.safe_load(f)
        dataset_config = all_configs.get("dataset_config") or all_configs.get("dataset") or {}
        baseline_config = all_configs.get("baseline_config") or all_configs.get("baseline_model") or {}
        self.initial_dataset_config = DatasetConfig.from_dict(dataset_config)
        self.baseline_config = BaselineConfig.from_dict(baseline_config)
        

    