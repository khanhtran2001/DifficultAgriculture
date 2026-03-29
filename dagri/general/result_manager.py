import os
import shutil
from pathlib import Path
import json
import yaml
from dataclasses import asdict, is_dataclass

from dagri.interfaces import DatasetProperties, BaselineProperties, EvaluationResults, PredictionResult, ScoringResults

class ResultManager:
    def __init__(self):
        pass

    @staticmethod
    def _prepare_output_file(output_dir: str | Path, file_name: str, overwrite: bool) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / file_name

        if output_file.exists():
            if overwrite:
                output_file.unlink()
            else:
                raise FileExistsError(
                    f"Output file already exists and overwrite=False: {output_file.resolve()}"
                )

        return output_file

    @staticmethod
    def _prepare_output_folder(output_dir: str | Path, overwrite: bool) -> Path:
        output_path = Path(output_dir)

        if output_path.exists():
            if overwrite:
                shutil.rmtree(output_path)
            else:
                raise FileExistsError(
                    f"Output directory already exists and overwrite=False: {output_path.resolve()}"
                )

        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def save_dataset_properties_to_json(
        self,
        output_dir: str,
        properties: DatasetProperties,
        overwrite: bool = True,
    ) -> None:
        # Backward compatibility: accept swapped argument order.
        if isinstance(output_dir, DatasetProperties) and isinstance(properties, (str, Path)):
            output_dir, properties = properties, output_dir

        output_file = self._prepare_output_file(output_dir, "dataset_properties.json", overwrite)

        if is_dataclass(properties):
            properties_dict = asdict(properties)
        else:
            properties_dict = dict(properties)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(properties_dict, f, indent=2)
        print(f"Dataset properties have been saved to: {output_file.resolve()}")

    def save_evaluation_results_to_json(
        self,
        output_dir: str,
        evaluation_results: EvaluationResults,
        file_name: str = "evaluation_results.json",
        overwrite: bool = True,
    ) -> None:
        """
        Save evaluation results JSON into an existing output directory.
        This method does not create any new folder.
        """
        output_file = self._prepare_output_file(output_dir, file_name, overwrite)

        if is_dataclass(evaluation_results):
            results_dict = asdict(evaluation_results)
        else:
            results_dict = dict(evaluation_results)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)
        print(f"Evaluation results have been saved to: {output_file.resolve()}")

    def save_prediction_results(
        self,
        output_dir: str,
        prediction_results: list[PredictionResult],
        file_format: str = "yolo_txt",
        overwrite: bool = True,
    ) -> None:
        """
        Save prediction results to files.
        
        Args:
            output_dir: Directory to save prediction files
            prediction_results: List of PredictionResult objects
            file_format: Output format (currently only 'yolo_txt' supported)
        """
        output_path = self._prepare_output_folder(output_dir, overwrite)
        
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
        print(f"Prediction results have been saved to: {output_path.resolve()} in format: {file_format}")

    def save_score_results_to_json(
        self,
        output_dir: str,
        score_results: ScoringResults,
        file_name: str = "score_results.json",
        overwrite: bool = True,
    ) -> None:
        """
        Save scoring output to JSON.
        """
        output_file = self._prepare_output_file(output_dir, file_name, overwrite)

        if is_dataclass(score_results):
            score_dict = asdict(score_results)
        else:
            score_dict = dict(score_results)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(score_dict, f, indent=2)

        print(f"Result for Scoring step has been saved to: {output_file.resolve()}")

    