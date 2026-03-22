"""
This file defines the interfaces for the DifficultyAgri project.
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.naive_bayes import abstractmethod


def _dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Return only keys that are declared in the target dataclass."""
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}

# Data Interface

@dataclass
class DatasetConfig:
    name: str = ""
    type: str = ""
    root_dir: str = ""
    train_mask_dir: Optional[str] = None
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        # Handle backward compatibility: map 'num_class' to 'num_classes'
        if data and "num_class" in data and "num_classes" not in data:
            data = {**data, "num_classes": data["num_class"]}
        return cls(**_dataclass_kwargs(cls, data or {}))

@dataclass
class DatasetProperties:
    root_dir: str = ""
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    train_mask_dir: Optional[str] = None
    train_images_dir: Optional[str] = None
    train_labels_dir: Optional[str] = None
    val_images_dir: Optional[str] = None
    val_labels_dir: Optional[str] = None
    test_images_dir: Optional[str] = None
    test_labels_dir: Optional[str] = None

@dataclass
class BaselineProperties:
    name: str
    model_type: str
    input_size: int
    traditional_augmentation: bool
    best_checkpoint_path: Optional[str]
    prediction_directory: Optional[str]

@dataclass
class TrainingConfig:
    input_size: int = 640
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    seed: int = 42
    early_stopping_patience: int = 20
    traditional_augmentation_config: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        data = data or {}
        # Handle augmentation -> traditional_augmentation_config mapping
        if "augmentation" in data and "traditional_augmentation_config" not in data:
            data = {**data, "traditional_augmentation_config": data["augmentation"]}
        return cls(**_dataclass_kwargs(cls, data))

@dataclass
class EvaluationConfig:
    save_dir: str = "runs/evaluation"
    image_size: Optional[int] = None
    confidence_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    max_detections: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        return cls(**_dataclass_kwargs(cls, data or {}))

@dataclass
class BaselineConfig:
    name: str = ""
    model_type: str = ""
    finetune_model_path: Optional[str] = None
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineConfig":
        data = data or {}
        
        # Handle field name mappings for backward compatibility
        if "pretrained_weights_path" in data and "finetune_model_path" not in data:
            data = {**data, "finetune_model_path": data["pretrained_weights_path"]}
        
        values = _dataclass_kwargs(cls, data)
        # Handle training_parameters -> training_config mapping
        training_data = data.get("training_config") or data.get("training_parameters") or {}
        values["training_config"] = TrainingConfig.from_dict(training_data)
        
        # Handle evaluation_parameters -> evaluation_config mapping
        evaluation_data = data.get("evaluation_config") or data.get("evaluation_parameters") or {}
        values["evaluation_config"] = EvaluationConfig.from_dict(evaluation_data)
        
        return cls(**values)




# Module Interfaces
# These are the interfaces for the main modules in the project.
# Each module should implement these interfaces to ensure consistency across the project.

class DatasetInterface:

    @abstractmethod
    def validate(self, output_dir: str) -> bool:
        """
        Validate the dataset and save the results and any relevant metadata to the specified output directory.
        """
        pass

    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """
        Get the properties of the dataset, such as the number of images, class distribution, etc.
        """
        pass

    @abstractmethod
    def save_results(self, output_dir: str) -> None:
        """
        Save the results of the dataset validation and any relevant metadata to the specified output directory.
        """
        pass


class BaselineInterface:
    @abstractmethod
    def train(self, training_config: dict) -> None:
        """
        Train the model based on the provided training configuration.
        """
        pass

    @abstractmethod
    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Predict the objects in the given image and return the results.
        """
        pass
    
    @abstractmethod
    def evaluate(self, evaluation_config: dict) -> dict:
        """
        Evaluate the model based on the provided evaluation configuration.
        """
        pass


class ScorerInterface:
    def score(self, scoring_config: dict) -> dict:
        """
        Score the model based on the provided scoring configuration.
        """
        pass


class AugmentorInterface:
    def create_new_train_dataset(self, augmentation_config: dict) -> dict:
        """
        Apply data augmentation based on the provided augmentation configuration.
        """
        pass
