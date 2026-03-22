import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from dagri.data.dataset import CustomDataset
from dagri.interfaces import DatasetProperties


class TestCustomDataset(unittest.TestCase):
    def test_initialize_from_config_creates_minneapple_yolo_dataset(self):
        config = {
            "name": "minneapple",
            "type": "yolo_detection",
            "root_dir": "dummy_root",
            "num_class": 1,
            "class_names": ["apple"],
        }

        with patch("dagri.data.dataset.MinneappleYoloDetectionDataset") as dataset_cls:
            created_dataset = object()
            dataset_cls.return_value = created_dataset

            custom_dataset = CustomDataset()
            custom_dataset.initialize_from_config(config)

            passed_config = dataset_cls.call_args.args[0]
            self.assertEqual(passed_config.name, "minneapple")
            self.assertEqual(passed_config.num_classes, 1)
            self.assertIs(custom_dataset.dataset, created_dataset)

    def test_initialize_from_config_raises_for_unsupported_dataset(self):
        custom_dataset = CustomDataset()

        with self.assertRaises(ValueError) as context:
            custom_dataset.initialize_from_config({"name": "other", "type": "classification"})

        self.assertIn("Unsupported dataset name", str(context.exception))

    def test_validate_delegates_to_underlying_dataset(self):
        custom_dataset = CustomDataset()
        custom_dataset.dataset = Mock()
        custom_dataset.dataset.validate.return_value = True

        result = custom_dataset.validate(output_dir="unused")

        custom_dataset.dataset.validate.assert_called_once_with()
        self.assertTrue(result)

    def test_get_properties_delegates_to_underlying_dataset(self):
        expected_properties = DatasetProperties(
            root_dir="/tmp/data",
            train_mask_dir="/tmp/masks",
            num_classes=1,
            class_names=["apple"],
            train_images_dir="/tmp/data/train/images",
            train_labels_dir="/tmp/data/train/labels",
            val_images_dir="/tmp/data/val/images",
            val_labels_dir="/tmp/data/val/labels",
            test_images_dir="/tmp/data/test/images",
            test_labels_dir="/tmp/data/test/labels",
        )

        custom_dataset = CustomDataset()
        custom_dataset.dataset = Mock()
        custom_dataset.dataset.get_properties.return_value = expected_properties

        result = custom_dataset.get_properties()

        custom_dataset.dataset.get_properties.assert_called_once_with()
        self.assertEqual(result, expected_properties)

    def test_save_results_writes_dataset_properties_json(self):
        properties = DatasetProperties(
            root_dir="/data/minneapple",
            train_mask_dir="/data/minneapple/masks",
            num_classes=1,
            class_names=["apple"],
            train_images_dir="/data/minneapple/train/images",
            train_labels_dir="/data/minneapple/train/labels",
            val_images_dir="/data/minneapple/val/images",
            val_labels_dir="/data/minneapple/val/labels",
            test_images_dir="/data/minneapple/test/images",
            test_labels_dir="/data/minneapple/test/labels",
        )

        custom_dataset = CustomDataset()
        custom_dataset.dataset = Mock()
        custom_dataset.dataset.get_properties.return_value = properties

        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_dataset.save_results(tmp_dir)

            output_file = os.path.join(tmp_dir, "dataset_properties.json")
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, "r", encoding="utf-8") as f:
                saved = json.load(f)

        self.assertEqual(
            saved,
            {
                "root_dir": "/data/minneapple",
                "train_mask_dir": "/data/minneapple/masks",
                "num_classes": 1,
                "class_names": ["apple"],
                "train_images_dir": "/data/minneapple/train/images",
                "train_labels_dir": "/data/minneapple/train/labels",
                "val_images_dir": "/data/minneapple/val/images",
                "val_labels_dir": "/data/minneapple/val/labels",
                "test_images_dir": "/data/minneapple/test/images",
                "test_labels_dir": "/data/minneapple/test/labels",
            },
        )


if __name__ == "__main__":
    unittest.main()
