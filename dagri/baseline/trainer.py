from pathlib import Path
from ultralytics import YOLO

class CustomTrainer:
    def __init__(self):
        self.model_path = Path("yolov8m.pt") 



class YoloTrainer(CustomTrainer):
    def train(self, epochs: int = 100, batch_size: int = 16):
        model = YOLO(self.model_path)
        model.train(data=self.data_path, epochs=epochs, batch=batch_size, save_dir=self.save_dir)


