from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.ssl.byol import BYOL
from src.models.vae.vae import VAE
from src.scripts.etl_process.ETLProcessor import ETLProcessor
from src.training.MaskDataset import MaskedDataset
from src.training.Trainer import Trainer


CONFIG = {
    "input_dim": 3,
    "hidden_dim": 128,
    "residual_hiddens": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mask_size": 0.35,
    "latent_dim": 1024,
    "param_grid": {"num_residual_layers": [1, 2]},
    "save_dir": Path("models/reconstruction/vae-byol/"),
    "dataset_config": {
        "kaggle_dataset": "mahmudulhaqueshawon/cat-image",
        "raw_dir": "data/raw_data",
        "split_dir": "data/data_splits",
    },
    "byol_models": [
        "models/ssl/byol/byol_l1_mlp2048_mask00_tau099.pt",
        "models/ssl/byol/byol_l1_mlp1024_mask035_tau097.pt",
        "models/ssl/byol/byol_l2_mlp512_mask035_tau099.pt",
        "models/ssl/byol/byol_l2_mlp1024_mask00_tau097.pt",
    ],
}

CONFIG["vis_dir"] = CONFIG["save_dir"] / "metrics"
CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["vis_dir"].mkdir(parents=True, exist_ok=True)


def extract_byol_encoders() -> dict:
    """
    Loads BYOL model encoders from specified model paths in CONFIG["byol_models"].
    Returns a list of encoder models with loaded weights.
    """
    encoders = {}
    for model_path in CONFIG["byol_models"]:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Model {model_path} does not exist.")
            continue

        num_residual_layers = int(model_path.stem.split("_")[1][1])
        mask = float(model_path.stem.split("_")[3][4:])
        mlp_hidden_dim = int(model_path.stem.split("_")[2][3:])

        byol_model = BYOL(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            residual_hiddens=CONFIG["residual_hiddens"],
            num_residual_layers=num_residual_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        state_dict = torch.load(model_path, map_location=CONFIG["device"])
        byol_model.load_state_dict(state_dict)

        encoder = byol_model.online_encoder
        encoder.to(CONFIG["device"])
        encoder.eval()

        encoders[
            "byol_vae_layers"
            + str(num_residual_layers)
            + "_maskedBYOL"
            + ("True" if mask > 0 else "False")
            + "_latent"
            + str(CONFIG["latent_dim"])
        ] = {
            "model": encoder,
            "num_residual_layers": num_residual_layers,
        }

    return encoders


def main():
    print("Preparing data...")
    print(f"Using device: {CONFIG['device']}")
    etl = ETLProcessor(**CONFIG["dataset_config"])
    train_loader, val_loader, _ = etl.process()
    masked_val_ds = MaskedDataset(val_loader.dataset, CONFIG["mask_size"])
    val_loader = DataLoader(
        masked_val_ds,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=val_loader.pin_memory,
        drop_last=val_loader.drop_last,
    )

    encoders = extract_byol_encoders()

    for model_name, params in encoders.items():
        print(f"Running: {model_name}")

        model = VAE(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            residual_hiddens=CONFIG["residual_hiddens"],
            num_residual_layers=params["num_residual_layers"],
            latent_dim=CONFIG["latent_dim"],
        )
        model = model.to(CONFIG["device"])
        byol_encoder = params["model"]

        model.encoder = byol_encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

        loss_fn = nn.MSELoss(reduction="sum")

        trainer = Trainer()
        trainer.train_supervised(
            model=model,
            epochs=CONFIG["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
            mask_ratio=CONFIG["mask_size"],
            loss_fn=loss_fn,
            model_name=model_name,
            save_dir=CONFIG["save_dir"],
            vis_dir=CONFIG["vis_dir"],
        )

    print("Training and visualization complete.")


if __name__ == "__main__":
    main()
