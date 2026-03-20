"""
This file defines the interfaces for the DifficultyAgri project.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.naive_bayes import abstractmethod


# ══════════════════════════════════════════════════════════════════════
# RAW DATA STRUCTURES
# These are the fundamental units that come directly from the dataset.
# Produced by: data/dataset.py
# Consumed by: scoring/, augmentation/, baseline/
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DatasetConfig:
    """
    Configuration for a dataset.
    """
    name: str
    root_dir: str 
    train_mask_dir: Optional[str] = None
    num_classes: int
    class_names: List[str]



# ══════════════════════════════════════════════════════════════════════
# Object Detection Baseline STRUCTURES
# These are the fundamental training and evaluation structures for the object detection baseline.
# Produced by: baseline/models.py
# Consumed by: scoring/
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.
    """
    name: str
    parameters: Dict[str, Any]


@dataclass
class ModelConfig:
    """
    Configuration for a model.
    """
    name: str
    pretrained: bool
    num_classes: int 
    input_size: int
    augmentation: AugmentationConfig

@dataclass
class TrainingConfig:
    """
    Configuration for training a model.
    """
    model_config: ModelConfig
    dataset_config: DatasetConfig
    epochs: int
    batch_size: int
    learning_rate: float
    early_stopping: bool
    early_stopping_patience: int
    save_dir: str

@dataclass
class EvaluationConfig:
    """
    Configuration for evaluating a model.
    """
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    max_detections: Optional[int] = None

@dataclass
class EvaluationResult:
    """
    Result of evaluating a model.
    """
    detection_results_directory: str
    best_model_weights_path: str
    mAP: float
    AP_50: float
    AP_75: float
    AP_small: float
    AP_medium: float
    AP_large: float


# ══════════════════════════════════════════════════════════════════════
# Scoring Structures
# These are the fundamental structures for scoring the models based on their performance.
# Produced by: scoring/scorer.py
# Consumed by: augmentation/
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ScoringConfig:
    """
    Configuration for scoring models.
    """
    name: str
    parameters: Dict[str, Any]

@dataclass
class ScoringResult:
    """
    Result of scoring a model.
    """
    model_name: str
    dataset_name: str
    image_score_dict: Dict[str, Any]
    object_score_dict: Dict[str, Any]

# ══════════════════════════════════════════════════════════════════════
# Augmentation Structures 
# These are the fundamental structures for data augmentation.
# Produced by: augmentation/augmentor.py
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.
    """
    name: str
    parameters: Dict[str, Any]


@dataclass
class AugmentationResult:
    """
    Result of applying data augmentation.
    """
    augmented_images_path: str
    augmented_labels_path: str


# Module Interfaces
# These are the interfaces for the main modules in the project.
# Each module should implement these interfaces to ensure consistency across the project.

class DatasetInterface:
    @abstractmethod
    def load_dataset(self, dataset_config: DatasetConfig) -> None:
        """
        Load the dataset based on the provided configuration.
        """
        pass

    @abstractmethod    
    def validate_dataset(self) -> bool:
        """
        Validate the dataset to ensure it meets the required format and structure.
        """
        pass

    @abstractmethod
    def get_data_config(self) -> DatasetConfig:
        """
        Return the dataset configuration.
        """
        pass


class ModelInterface:
    @abstractmethod
    def train(self, training_config: TrainingConfig) -> EvaluationResult:
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
    def evaluate(self, evaluation_config: EvaluationConfig) -> EvaluationResult:
        """
        Evaluate the model based on the provided evaluation configuration.
        """
        pass


class ScorerInterface:
    def score(self, scoring_config: ScoringConfig) -> ScoringResult:
        """
        Score the model based on the provided scoring configuration.
        """
        pass


class AugmentorInterface:
    def create_new_train_dataset(self, augmentation_config: AugmentationConfig) -> AugmentationResult:
        """
        Apply data augmentation based on the provided augmentation configuration.
        """
        pass
