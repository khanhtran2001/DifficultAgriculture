import os
import json

from dagri.interfaces import DatasetInterface, DatasetConfig, DatasetProperties
from dagri.data.minneapple import MinneappleYoloDetectionDataset
from dagri.data.wheathead import WheatHeadYoloDetectionDataset

class CustomDataset(DatasetInterface):
    def __init__(self, dataset_config: DatasetConfig | dict) -> None:
        """
        Initialize the dataset from typed config or plain dictionary.
        """
        if isinstance(dataset_config, dict):
            dataset_config = DatasetConfig.from_dict(dataset_config)

        dataset_name = (dataset_config.name or "").strip().lower()
        if dataset_name in {"minneapple", "global_wheat_head"} and dataset_config.type == "yolo_detection":
            self.dataset = MinneappleYoloDetectionDataset(dataset_config)
        elif dataset_name in {"wheat_head"} and dataset_config.type == "yolo_detection":
            self.dataset = WheatHeadYoloDetectionDataset(dataset_config)
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name} or type: {dataset_config.type}   ")

    def validate(self) -> None:
        """
        Validate the dataset and save the results and any relevant metadata to the specified output directory.
        """
        return self.dataset.validate()

    def get_properties(self) -> DatasetProperties:
        """
        Get the properties of the dataset, such as the number of images, class distribution, etc.
        """
        return self.dataset.get_properties()

    
