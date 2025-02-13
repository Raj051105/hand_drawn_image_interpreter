import torch
import torchvision
import yaml
import os
from ultralytics import YOLO

def train_yolo():
    """
    Trains a YOLOv8 model on the FloCo dataset.
    """
    dataset_path = "./"
    train_path = os.path.join(dataset_path, "Train")
    val_path = os.path.join(dataset_path, "Validation")
    
    # Define dataset.yaml file for YOLO
    dataset_yaml = {
        'path': dataset_path,
        'train': 'Train/png',
        'val': 'Validation/png',
        'test': 'Test/png',
        'names': {0: 'start_end', 1: 'process', 2: 'decision', 3: 'connector', 4: 'input_output'}
    }
    
    # Save dataset.yaml file
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f)
    
    # Initialize and train YOLO model
    model = YOLO("yolov8n.pt")  # Use a pre-trained YOLOv8 model
    model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=16,
        device='cuda',
        name='yolo_floco_model'
    )
    
    # Save final model
    model.export(format='torchscript')
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train_yolo()
