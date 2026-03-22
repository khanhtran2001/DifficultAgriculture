import os
from pathlib import Path
import json
import yaml
from dataclasses import asdict, is_dataclass

from dagri.interfaces import DatasetProperties, BaselineProperties 

class ResultManager:
    def __init__(self):
        pass

    def save_dataset_properties_to_json(self,output_dir: str, properties: DatasetProperties) -> None:
        # Backward compatibility: accept swapped argument order.
        if isinstance(output_dir, DatasetProperties) and isinstance(properties, (str, Path)):
            output_dir, properties = properties, output_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if is_dataclass(properties):
            properties_dict = asdict(properties)
        else:
            properties_dict = dict(properties)

        with open(output_path / "dataset_properties.json", "w", encoding="utf-8") as f:
            json.dump(properties_dict, f, indent=2)

    