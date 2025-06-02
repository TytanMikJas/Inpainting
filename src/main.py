import sys
import torch
import torch.nn as nn
from src.scripts.etl_process.ETLProcessor import ETLProcessor
from src.training.TrainScheduler import TrainScheduler, SingleTrainEvent
from src.models.vae.VAE import VAE


def run_etl():
    DATASET_ID = "mahmudulhaqueshawon/cat-image"
    DATA_DIR = "data/raw_data"
    SPLIT_DATA_DIR = "data/data_splits"

    etl_processor: ETLProcessor = ETLProcessor(DATASET_ID, DATA_DIR, SPLIT_DATA_DIR)

    print("Starting ETL process...")
    etl_processor.process()
    print("ETL process completed successfully.")


def run_training():
    print("Running training...")

    etl = ETLProcessor(
        kaggle_dataset="mahmudulhaqueshawon/cat-image",
        raw_dir="data/raw_data",
        split_dir="data/data_splits",
    )
    train_loader, val_loader, test_loader = etl.process()

    model = VAE(
        input_dim=3,
        hidden_dim=128,
        residual_hiddens=64,
        num_residual_layers=2,
        latent_dim=256,
    )

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = nn.MSELoss(reduction="sum")

    events = [
        SingleTrainEvent(
            epochs=50,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=1e-3,
            loss_fn=loss_fn,
            loss_fn_args=None,
            mask_size=0.35,
            save_dir="models/reconstruction/vae",
        ),
    ]

    scheduler = TrainScheduler(events, model, model_name="test_vae")
    scheduler.start_training()

    print("Training completed.")


if __name__ == "__main__":
    command = sys.argv[1]
    print(f"Command received: {command}")
    if command == "run_etl":
        run_etl()
    elif command == "run_training":
        run_training()
    else:
        print(
            "Usage: python main.py <command>\nCommands:\netl - Run ETL process\nrun_training - Run training"
        )
        sys.exit(1)
