import os

from dagri.interfaces import  DatasetProperties, DatasetConfig

class MinneappleYoloDetectionDataset():
    """
    Dataset class for the Minneapple dataset in YOLO format for object detection tasks.
    This class implements the DatasetInterface and provides methods for loading, validating, and retrieving properties of the Minneapple dataset.
    The dataset is expected to be organized in a specific directory structure with separate folders for training, validation, and testing images and labels, as well as optional mask images and labels for augmentation purposes.
    """

    def __init__(self, dataset_config: DatasetConfig):
        """
        Initialize the MinneappleYoloDetectionDataset with the provided dataset configuration.
        """
        self.dataset_config = dataset_config
        self.dataset_properties = self._extract_properties_from_config(dataset_config)
        

    def _extract_properties_from_config(self, dataset_config: DatasetConfig) -> DatasetProperties:
        """
        Extract the dataset properties from the provided dataset configuration.
        """
         # Define the expected directory structure based on the root directory and standard subdirectories for YOLO format
        self.train_images_dir = os.path.join(dataset_config.root_dir, "train/images")
        self.train_labels_dir = os.path.join(dataset_config.root_dir, "train/labels")
        self.val_images_dir = os.path.join(dataset_config.root_dir, "val/images")
        self.val_labels_dir = os.path.join(dataset_config.root_dir, "val/labels")
        self.test_images_dir = os.path.join(dataset_config.root_dir, "test/images")
        self.test_labels_dir = os.path.join(dataset_config.root_dir, "test/labels")

        return DatasetProperties(
            root_dir=dataset_config.root_dir,
            train_mask_dir=dataset_config.train_mask_dir,
            num_classes=dataset_config.num_classes,
            class_names=dataset_config.class_names,
            train_images_dir=self.train_images_dir,
            train_labels_dir=self.train_labels_dir,
            val_images_dir=self.val_images_dir,
            val_labels_dir=self.val_labels_dir,
            test_images_dir=self.test_images_dir,
            test_labels_dir=self.test_labels_dir,
        )
        
    def validate(self) -> bool:
        """
        Validate the dataset and return a validation report as a dictionary.
        """
        if not os.path.exists(self.dataset_config.root_dir):
            raise FileNotFoundError(f"Dataset root directory {self.dataset_config.root_dir} does not exist.")
        # Check if all required directories exist
        required_dirs = [
            self.train_images_dir,
            self.train_labels_dir,
            self.val_images_dir,
            self.val_labels_dir,
            self.test_images_dir,
            self.test_labels_dir
        ]
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory {dir_path} does not exist.")
        # Check if the number and name of images matches the number and name of labels in each split
        for images_dir, labels_dir in [
            (self.train_images_dir, self.train_labels_dir),
            (self.val_images_dir, self.val_labels_dir),
            (self.test_images_dir, self.test_labels_dir)
        ]:
            # Compare base filenames without extensions (images are .png/.jpg, labels are .txt)
            image_bases = {os.path.splitext(f)[0] for f in os.listdir(images_dir)}
            label_bases = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)}
            if image_bases != label_bases:
                raise ValueError(f"Mismatch between images and labels in {images_dir} and {labels_dir}.")
        return True
         
        
    def get_properties(self) -> DatasetProperties:
        """
        Get the properties of the dataset, such as the number of images, class distribution, etc.
        """
        return self.dataset_properties
    
    
