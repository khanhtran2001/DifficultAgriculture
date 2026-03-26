from dagri.interfaces import BaselineConfig, DatasetProperties
from dagri.baseline.yolo_model import YoloUltralyticsModel


class Baseline:
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
    
    def custom_evaluate(self, best_weight_path: str, dataset_properties: DatasetProperties, output_dir: str) -> dict:
        """Backward-compatible alias for test-set evaluation."""
        return self.model.custom_evaluate(best_weight_path, dataset_properties, output_dir)
    
    def custom_predict(self, model_weight: str, image_path: str, output_dir: str, conf: float, iou: float, max_det: int) -> None:
        """
        Run inference on a single image.
        """
        return self.model.custom_predict(model_weight, image_path, output_dir, conf, iou, max_det)






