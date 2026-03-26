from dagri.interfaces import BaselineConfig, DatasetProperties, EvaluationResults, BaselineInterface, PredictionResult
from dagri.baseline.yolo_model import YoloUltralyticsModel


class Baseline(BaselineInterface):
    """
    Baseline model wrapper that routes to concrete implementations.
    Accepts typed config and dataset properties.
    """
    def __init__(self, baseline_config: BaselineConfig | dict) -> None:
        """
        Initialize the baseline model from typed config and dataset properties.
        
        Args:
            baseline_config: Baseline model configuration (typed or dict)
            dataset_properties: Dataset properties containing train/val/test directories and class names.
        """
        if isinstance(baseline_config, dict):
            baseline_config = BaselineConfig.from_dict(baseline_config)
        
        model_type = baseline_config.model_type
        if model_type == "yolo":
            self.model = YoloUltralyticsModel(baseline_config)
        else:
            raise ValueError(f"Unsupported baseline model type: {model_type}")
    
    def custom_train(self, dataset_properties: DatasetProperties, output_dir: str) -> str:
        """
        Train the baseline model and return path to best weights.
        """
        return self.model.custom_train(dataset_properties, output_dir)
    
    def custom_evaluate_on_test_set(self, best_weight_path: str, dataset_properties: DatasetProperties) -> EvaluationResults:
        """Evaluate baseline model on test set and return typed evaluation results."""
        return self.model.custom_evaluate_on_test_set(best_weight_path, dataset_properties)

    
    def custom_predict(self, model_weight: str, image_dir: str, conf: float, iou: float, max_det: int) -> list[PredictionResult]:
        """
        Run inference on a directory of images and return prediction results.
        """
        return self.model.custom_predict(model_weight, image_dir, conf, iou, max_det)

    def get_optimal_conf_threshold_for_scoring(
        self,
        dataset_properties: DatasetProperties,
        model_weight: str,
        conf_min: float = 0.001,
        conf_max: float = 0.9,
        num_points: int = 20,
    ) -> float:
        """
        Run a grid search on validation set to find the best confidence threshold.
        """
        return self.model.get_optimal_conf_threshold_for_scoring(
            dataset_properties=dataset_properties,
            model_weight=model_weight,
            conf_min=conf_min,
            conf_max=conf_max,
            num_points=num_points,
        )






