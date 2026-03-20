from dagri.interfaces import DatasetInterface

class YoloTypeDataset(DatasetInterface):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path 
    
    def load_dataset(self, dataset_config) -> None:
        # Implement dataset loading logic here
        pass

    def get_data_config(self):
        # Implement logic to return dataset configuration
        pass

    def validate_dataset(self) -> bool:
        pass 
