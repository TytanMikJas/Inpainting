import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scripts.etl_process.ETLProcessor import ETLProcessor
from scripts.training.TrainScheduler import TrainScheduler, SingleTrainEvent
from models.vae.vae import VAE


def run_etl():
    DATASET_ID = "mahmudulhaqueshawon/cat-image"
    DATA_DIR = "data/raw_data"
    SPLIT_DATA_DIR = "data/data_splits"

    etl_processor: ETLProcessor = ETLProcessor(DATASET_ID, DATA_DIR, SPLIT_DATA_DIR)

    print("Starting ETL process...")
    etl_processor.process()
    print("ETL process completed successfully.")


def run_dummy_training():
    print("Running dummy training...")

    num_samples = 100
    images = torch.randn(num_samples, 3, 64, 64)
    dataset = TensorDataset(images)

    batch_size = 16
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = VAE(
        input_dim=3,
        hidden_dim=32,
        residual_hiddens=32,
        num_residual_layers=2,
        latent_dim=16,
    )

    loss_fn = nn.MSELoss(reduction="sum")

    events = [
        SingleTrainEvent(
            epochs=1,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=1e-3,
            loss_fn=loss_fn,
            loss_fn_args=None,
            noise_level=0.1,
            save_dir="models/test_run",
        ),
        SingleTrainEvent(
            epochs=1,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=1e-3,
            loss_fn=loss_fn,
            loss_fn_args=None,
            noise_level=0.2,
            save_dir="models/test_run",
        ),
    ]

    scheduler = TrainScheduler(events, model, model_name="test_vae")
    scheduler.start_training()

    print("Dummy training completed.")


if __name__ == "__main__":
    command = sys.argv[1]
    print(f"Command received: {command}")
    if command == "run_etl":
        run_etl()
    elif command == "run_dummy_training":
        run_dummy_training()
    else:
        print(
            "Usage: python main.py <command>\nCommands:\netl - Run ETL process\nrun_dummy_training - Run dummy training"
        )
        sys.exit(1)
