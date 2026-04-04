"""
Augmentation-Only Experiment

This experiment focuses solely on the augmentation step. It loads pre-computed results
(dataset properties and scoring results) from a previous full experiment, then runs
the copy-paste augmentation to create a new augmented dataset.

This avoids duplicate training time by reusing existing baseline model results.

Usage:
    python experiments/09_aug_exp_only.py --source-exp 07_minneapple_yolo_full_3_seed --seed 123
"""

import argparse
import json
from pathlib import Path
import sys
from dataclasses import asdict

# Allow running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dagri.general.config_manager import ConfigManager
from dagri.general.result_manager import ResultManager
from dagri.interfaces import DatasetProperties, ScoringResults, ImageDifficultyProperties, ObjectDifficultyProperties
from dagri.augmentation import CopyPasteAugmentor

import experiments.utils as exputils


RESULTS_DIR = Path(f"results/{Path(__file__).stem}")
CONFIG_DIR = Path("/home/khanh/Projects/DifficultyAgri/configs/experiments/minneapple_yolo.yaml")


def load_dataset_properties(dataset_properties_file: str | Path) -> DatasetProperties:
    """Load dataset properties from JSON file."""
    with open(dataset_properties_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetProperties(**data)


def load_scoring_results(scoring_results_file: str | Path) -> ScoringResults:
    """Load scoring results from JSON file."""
    with open(scoring_results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Reconstruct nested dataclass structures
    image_difficulties = []
    for img_data in data.get("image_difficulties", []):
        objects_score = [
            ObjectDifficultyProperties(**obj) for obj in img_data.get("objects_score", [])
        ]
        img_data["objects_score"] = objects_score
        image_difficulties.append(ImageDifficultyProperties(**img_data))
    
    data["image_difficulties"] = image_difficulties
    return ScoringResults(**data)


def run_augmentation_only_experiment(
    source_exp: str,
    seed: int,
    config_path: str | Path = CONFIG_DIR,
    results_base_dir: Path = RESULTS_DIR,
) -> None:
    """
    Run augmentation-only experiment using results from a previous full experiment.
    
    Args:
        source_exp: Name of the source experiment (e.g., "07_minneapple_yolo_full_3_seed")
        seed: Random seed to use
        config_path: Path to configuration YAML file
        results_base_dir: Base directory for results
    """
    
    # Setup output directory
    results_base_dir.mkdir(parents=True, exist_ok=True)
    frozen_config_path = results_base_dir / "frozen_config.yaml"
    exputils.copy_yaml_config(config_path, frozen_config_path)
    
    # Load configuration
    config_manager = ConfigManager()
    config_manager.load_all_configs(config_path)
    augmentation_config = config_manager.augmentation_config
    
    result_manager = ResultManager()
    
    # Determine source paths
    source_exp_dir = Path("results") / source_exp / f"seed_{seed}"
    step_1_dir = source_exp_dir / "Step_1_Load_and_Validate_Dataset"
    step_3_dir = source_exp_dir / "Step_3_Scoring_Dataset"
    
    # Verify source directories exist
    if not step_1_dir.exists():
        raise FileNotFoundError(f"Source dataset properties directory not found: {step_1_dir}")
    if not step_3_dir.exists():
        raise FileNotFoundError(f"Source scoring results directory not found: {step_3_dir}")
    
    dataset_properties_file = step_1_dir / "dataset_properties.json"
    scoring_results_file = step_3_dir / "score_results.json"
    
    if not dataset_properties_file.exists():
        raise FileNotFoundError(f"Dataset properties file not found: {dataset_properties_file}")
    if not scoring_results_file.exists():
        raise FileNotFoundError(f"Scoring results file not found: {scoring_results_file}")
    
    print(f"\n{'='*70}")
    print(f"Augmentation-Only Experiment")
    print(f"{'='*70}")
    print(f"Loading from source experiment: {source_exp} (seed: {seed})")
    print(f"Dataset properties: {dataset_properties_file}")
    print(f"Scoring results: {scoring_results_file}")
    print(f"Output directory: {results_base_dir}")
    
    # Load dataset properties and scoring results
    print("\n[Step 1] Loading dataset properties...")
    initial_dataset_properties = load_dataset_properties(dataset_properties_file)
    print(f"  ✓ Loaded dataset with {initial_dataset_properties.num_classes} classes")
    print(f"  ✓ Train images: {initial_dataset_properties.train_images_dir}")
    print(f"  ✓ Train labels: {initial_dataset_properties.train_labels_dir}")
    
    print("\n[Step 2] Loading scoring results...")
    score_results = load_scoring_results(scoring_results_file)
    print(f"  ✓ Loaded scoring results")
    print(f"  ✓ Scoring mode: {score_results.scoring_weight_mode}")
    print(f"  ✓ Object weight: {score_results.selected_object_weight:.4f}")
    print(f"  ✓ False positive weight: {score_results.selected_false_positive_weight:.4f}")
    print(f"  ✓ Images scored: {len(score_results.image_difficulties)}")
    
    # Run augmentation
    print("\n[Step 3] Running copy-paste augmentation...")
    seed_output_dir = results_base_dir / f"seed_{seed}"
    step_4_dir = seed_output_dir / "Step_4_Copy_Paste_Augmentation"
    step_4_dir.mkdir(parents=True, exist_ok=True)
    
    augmentor = CopyPasteAugmentor(augmentation_config)
    augmented_dataset_dir = step_4_dir / "augmented_dataset"
    
    print(f"  Creating augmented dataset at: {augmented_dataset_dir}")
    new_dataset_properties = augmentor.create_new_dataset(
        initial_dataset_properties=initial_dataset_properties,
        scoring_results=score_results,
        new_dataset_path=augmented_dataset_dir,
    )
    print(f"  ✓ Augmentation complete")
    
    # Save augmented dataset properties
    print("\n[Step 4] Saving results...")
    result_manager.save_dataset_properties_to_json(step_4_dir, new_dataset_properties)
    
    # Save summary
    summary = {
        "source_experiment": source_exp,
        "source_seed": int(seed),
        "augmentation_config": asdict(augmentation_config) if hasattr(augmentation_config, '__dataclass_fields__') else dict(augmentation_config),
        "scoring_mode": score_results.scoring_weight_mode,
        "object_weight": float(score_results.selected_object_weight),
        "false_positive_weight": float(score_results.selected_false_positive_weight),
        "original_dataset_train_images": initial_dataset_properties.train_images_dir,
        "original_dataset_train_labels": initial_dataset_properties.train_labels_dir,
        "augmented_dataset_path": str(augmented_dataset_dir),
        "augmented_dataset_train_count": len([f for f in (augmented_dataset_dir / "images" / "train").glob("*") if f.is_file()]) if (augmented_dataset_dir / "images" / "train").exists() else 0,
    }
    
    summary_path = seed_output_dir / "augmentation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ Summary saved to: {summary_path}")
    print(f"\n{'='*70}")
    print(f"✓ Augmentation-only experiment completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run augmentation-only experiment using results from previous full experiments"
    )
    parser.add_argument(
        "--source-exp",
        type=str,
        default="07_minneapple_yolo_full_3_seed",
        help="Name of source experiment directory (default: 07_minneapple_yolo_full_3_seed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed to use from source experiment (default: 123)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_DIR),
        help="Path to configuration YAML file",
    )
    
    args = parser.parse_args()
    
    run_augmentation_only_experiment(
        source_exp=args.source_exp,
        seed=args.seed,
        config_path=args.config,
        results_base_dir=RESULTS_DIR,
    )
