import torch
import wandb
from ultralytics import YOLO


data_yaml = "Branch Dataset.v1i.yolo26/data.yaml"

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

    with wandb.init(project="yolo26_branch", name="branch") as run:

            # Initialize YOLO Model
            model = YOLO('yolo26n.pt')

            # Train/fine-tune your model
            results = model.train(
                data=str(data_yaml),
                name='branch',
                **train_params,
                mosaic=0.0,
                mixup=0.0,
            )
    wandb.finish()


if __name__ == "__main__":
    main()