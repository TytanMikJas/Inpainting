import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.ssl.byol import BYOL
from src.models.ssl.barlow_twins.BarlowTwins import BarlowTwins
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
    "dataset_config": {
        "kaggle_dataset": "mahmudulhaqueshawon/cat-image",
        "raw_dir": "data/raw_data",
        "split_dir": "data/data_splits",
    },
    "pretrained_models": {
        "byol": [
            "models/ssl/byol/byol_l1_mlp2048_mask00_tau099.pt",
            "models/ssl/byol/byol_l1_mlp1024_mask035_tau097.pt",
            "models/ssl/byol/byol_l2_mlp512_mask035_tau099.pt",
            "models/ssl/byol/byol_l2_mlp1024_mask00_tau097.pt",
        ],
        "barlow_twins": [
            "models/ssl/barlow_twins/barlow_twins_num_residual_layers1_mlp_hidden_dim512_mask_size_ratio035_lambda_0005.pt",
            "models/ssl/barlow_twins/barlow_twins_num_residual_layers1_mlp_hidden_dim512_mask_size_ratio035_lambda_001.pt",
            "models/ssl/barlow_twins/barlow_twins_num_residual_layers1_mlp_hidden_dim512_mask_size_ratio035_lambda_0001.pt",
        ],
    },
    "save_root": Path("models/reconstruction/vae-ssl/"),
}

CONFIG["save_root"].mkdir(parents=True, exist_ok=True)


def extract_encoders(method: str) -> dict:
    encoders = {}
    model_paths = CONFIG["pretrained_models"].get(method)
    if not model_paths:
        raise ValueError(f"No pretrained models found for method '{method}'.")

    for path_str in model_paths:
        model_path = Path(path_str)
        if not model_path.exists():
            print(f"Warning: Model {model_path} not found.")
            continue

        stem = model_path.stem
        if method == "byol":
            layers = int(stem.split("_")[1][1])
            mlp = int(stem.split("_")[2][3:])
            mask = float(stem.split("_")[3][4:])
            model = BYOL(
                input_dim=CONFIG["input_dim"],
                hidden_dim=CONFIG["hidden_dim"],
                residual_hiddens=CONFIG["residual_hiddens"],
                num_residual_layers=layers,
                mlp_hidden_dim=mlp,
            )
            encoder = model.online_encoder
        elif method == "barlow_twins":
            parts = stem.split("_")
            layers = int(parts[4].replace("layers", ""))
            mlp = int(parts[7].replace("dim", ""))
            mask = float("0." + parts[10].replace("ratio0", ""))
            lambda_ = float("0." + parts[12][1:])

            model = BarlowTwins(
                input_dim=CONFIG["input_dim"],
                hidden_dim=CONFIG["hidden_dim"],
                residual_hiddens=CONFIG["residual_hiddens"],
                num_residual_layers=layers,
                mlp_hidden_dim=mlp,
                lambda_=lambda_,
            )
            encoder = model.encoder
        else:
            raise ValueError(f"Unsupported method: {method}")

        state_dict = torch.load(model_path, map_location=CONFIG["device"])
        model.load_state_dict(state_dict)
        encoder = model.online_encoder if method == "byol" else model.encoder
        encoder.to(CONFIG["device"])
        encoder.eval()

        name = f"{method}_vae_l{layers}_masked{mask > 0}_latent{CONFIG['latent_dim']}"
        encoders[name] = {"model": encoder, "num_residual_layers": layers}

    return encoders


def main():
    if len(sys.argv) < 2:
        print("Usage: python finetune.py [byol|barlow_twins]")
        sys.exit(1)

    method = sys.argv[1]
    if method not in CONFIG["pretrained_models"]:
        print(f"Unsupported method: {method}")
        sys.exit(1)

    print(f"Preparing data using method: {method}")
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

    encoders = extract_encoders(method)
    save_dir = CONFIG["save_root"] / method
    vis_dir = save_dir / "metrics"
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for model_name, params in encoders.items():
        print(f"\nFine-tuning: {model_name}")
        vae = VAE(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            residual_hiddens=CONFIG["residual_hiddens"],
            num_residual_layers=params["num_residual_layers"],
            latent_dim=CONFIG["latent_dim"],
        ).to(CONFIG["device"])

        vae.encoder = params["model"]
        for param in vae.encoder.parameters():
            param.requires_grad = False

        trainer = Trainer()
        trainer.train_supervised(
            model=vae,
            epochs=CONFIG["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
            mask_ratio=CONFIG["mask_size"],
            loss_fn=nn.MSELoss(reduction="sum"),
            model_name=model_name,
            save_dir=save_dir,
            vis_dir=vis_dir,
        )

    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()
