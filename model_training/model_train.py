import torch
import wandb
import time
from ultralytics import YOLO


data_yaml = "/home/baraa/Desktop/Diplomski/Diplomski_rad/model_training/Branch_SAM_Segmentation/data.yaml"

def main():

    train_params = {
        'epochs': 200,
        'imgsz': 640,
        'batch': 8,
        'workers': 2,
        'cache': False,
        'save_period': 10,
        'plots': True,
        'patience': 100,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.002,
        'weight_decay': 0.0005,
    }

    start_time_cropped = time.time()
    with wandb.init(project="yolo26_branch", name="branch_segmentation") as run:

            # Initialize YOLO Model
            model = YOLO('yolo26n-seg.pt')

            # Train/fine-tune your model
            results = model.train(
                data=str(data_yaml),
                name='branch_segmentation',
                **train_params,
                mosaic=0.7,
                mixup=0.0,
            )
            time_cropped = time.time() - start_time_cropped
            wandb.log({"train_results": results, "train_time": time_cropped})
    wandb.finish()

    


if __name__ == "__main__":
    main()