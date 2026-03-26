"""
This file defines the interfaces for the DifficultyAgri project.
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


def _dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Return only keys that are declared in the target dataclass."""
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}

# Data Interface

"""
Data classes for configuration and properties of datasets, baselines, training, evaluation, and scoring.
These data classes provide a structured way to represent the various configurations and properties used throughout the project.
"""
@dataclass
class DatasetConfig:
    name: str
    type: str
    root_dir: str
    num_classes: int
    class_names: List[str]
    train_mask_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        data = dict(data or {})
        return cls(**_dataclass_kwargs(cls, data))

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

"""
Baseline properties and configuration data classes for representing the baseline model's configuration, training parameters, and evaluation parameters.
These data classes provide a structured way to represent the baseline model's configuration and properties used throughout the project.
"""

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
    epochs: int
    batch_size: int
    input_size: int
    epochs: int 
    batch_size: int 
    learning_rate: float 
    seed: int 
    early_stopping_patience: int
    traditional_augmentation_config: dict 

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        data = data or {}
        # Extract augmentation if present and nested
        aug_config = data.get("augmentation", {})
        # Get allowed fields from dataclass
        values = _dataclass_kwargs(cls, data)
        # Set augmentation config from nested augmentation key
        if aug_config:
            values["traditional_augmentation_config"] = aug_config
        return cls(**values)

@dataclass
class EvaluationConfig:
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    max_detections: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        data = data or {}
        values = _dataclass_kwargs(cls, data)
        return cls(**values)

@dataclass
class BaselineConfig:
    name: str
    model_type: str
    pretrained_weights_path: str
    training_config: TrainingConfig
    evaluation_config: EvaluationConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineConfig":
        data = data or {}
        values = _dataclass_kwargs(cls, data)
        # Look for training_config or training_parameters (YAML uses training_parameters)
        training_cfg = data.get("training_config") or data.get("training_parameters")
        evaluation_cfg = data.get("evaluation_config") or data.get("evaluation_parameters")
        if training_cfg is None:
            raise ValueError("Missing required 'training_config' or 'training_parameters' in baseline config")
        if evaluation_cfg is None:
            raise ValueError("Missing required 'evaluation_config' or 'evaluation_parameters' in baseline config")
        values["training_config"] = TrainingConfig.from_dict(training_cfg)
        values["evaluation_config"] = EvaluationConfig.from_dict(evaluation_cfg)
        return cls(**values)

@dataclass
class EvaluationResults:
    COCO_AP: float
    COCO_AP50: float
    COCO_AP75: float
    AP_small: float
    AP_medium: float
    AP_large: float

@dataclass
class PredictionResult:
    image_path: str
    classes: List[int]  # List of predicted class indices
    confidences: List[float]  # List of confidence scores for each prediction
    predicted_boxes: List[BoundingBox]  # Bounding box coordinates for each prediction

@dataclass
class BoundingBox:
    x_center: Optional[float] = None  # x_center coordinates (normalized 0-1)
    y_center: Optional[float] = None  # y_center coordinates (normalized 0-1)
    width: Optional[float] = None  # widths (normalized 0-1)
    height: Optional[float] = None  # heights (normalized 0-1)
    x_min: Optional[float] = None  # x_min coordinates (normalized 0-1)
    y_min: Optional[float] = None  # y_min coordinates (normalized 0-1)
    x_max: Optional[float] = None  #  x_max coordinates (normalized 0-1)
    y_max: Optional[float] = None  # y_max coordinates (normalized 0-1)


"""
Scoring configuration data class for representing the scoring configuration used to score the model's performance.
This data class provides a structured way to represent the scoring configuration used throughout the project.
"""

@dataclass
class ScoringConfig:
    type: str
    alpha: float 
    beta: float
    iou_threshold: float
    object_weight: float
    false_positive_weight: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringConfig":
        data = dict(data or {})
        values = _dataclass_kwargs(cls, data)

        required = [f.name for f in fields(cls)]
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(
                "Missing required scoring config field(s): " + ", ".join(missing)
            )
        return cls(**values)


@dataclass
class ScoringResults:
    image_difficulties: List[ImageDifficultyProperties]

@dataclass 
class ImageDifficultyProperties:
    image_path: str
    difficulty_score: float
    num_objects: int
    objects_score: List[ObjectDifficultyProperties]
    false_positive_rate: float
    missed_detections_rate: float

@dataclass
class ObjectDifficultyProperties:
    image_path: str
    object_id: int
    class_id: int
    bounding_box: BoundingBox
    difficulty_score: float



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
    def custom_train(self, training_config: dict) -> None:
        """
        Train the model based on the provided training configuration.
        """
        pass

    @abstractmethod
    def custom_predict(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Predict the objects in the given image and return the results.
        """
        pass
    
    @abstractmethod
    def custom_evaluate_on_test_set(self, evaluation_config: dict) -> dict:
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
