import os
from pathlib import Path
import json
import yaml
from dataclasses import asdict, is_dataclass

from dagri.interfaces import DatasetProperties, BaselineProperties, EvaluationResults, PredictionResult, ScoringResults

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

    def save_evaluation_results_to_json(
        self,
        output_dir: str,
        evaluation_results: EvaluationResults,
        file_name: str = "evaluation_results.json",
    ) -> None:
        """
        Save evaluation results JSON into an existing output directory.
        This method does not create any new folder.
        """
        output_path = Path(output_dir)
        if not output_path.exists() or not output_path.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {output_dir}. "
                "Create it beforehand; save_evaluation_results_to_json does not create folders."
            )

        if is_dataclass(evaluation_results):
            results_dict = asdict(evaluation_results)
        else:
            results_dict = dict(evaluation_results)

        with open(output_path / file_name, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)

    def save_prediction_results(
        self,
        output_dir: str,
        prediction_results: list[PredictionResult],
        file_format: str = "yolo_txt",
    ) -> None:
        """
        Save prediction results to files.
        
        Args:
            output_dir: Directory to save prediction files
            prediction_results: List of PredictionResult objects
            file_format: Output format (currently only 'yolo_txt' supported)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if file_format == "yolo_txt":
            for pred_result in prediction_results:
                image_name = Path(pred_result.image_path).stem
                output_file = output_path / f"{image_name}.txt"
                
                with open(output_file, "w", encoding="utf-8") as f:
                    for i, bbox in enumerate(pred_result.predicted_boxes):
                        cls = pred_result.classes[i]
                        conf = pred_result.confidences[i]
                        f.write(f"{cls} {bbox.x_center:.6f} {bbox.y_center:.6f} {bbox.width:.6f} {bbox.height:.6f} {conf:.6f}\n")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def save_score_results_to_json(
        self,
        output_dir: str,
        score_results: ScoringResults,
        file_name: str = "score_results.json",
    ) -> None:
        """
        Save scoring output to JSON.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if is_dataclass(score_results):
            score_dict = asdict(score_results)
        else:
            score_dict = dict(score_results)

        with open(output_path / file_name, "w", encoding="utf-8") as f:
            json.dump(score_dict, f, indent=2)

    