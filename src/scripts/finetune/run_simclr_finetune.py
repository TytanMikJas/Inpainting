from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.ssl.simclr.SimCLR import SimCLR
from src.models.vae.vae import VAE
from src.scripts.etl_process.ETLProcessor import ETLProcessor
from src.training.MaskDataset import MaskedDataset
from src.training.Trainer import Trainer

SSL_PREFIX = "models/ssl/simclr/"

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
    "param_grid": {"num_residual_layers": [1]},
    "save_dir": Path("models/reconstruction/vae-simclr/"),
    "dataset_config": {
        "kaggle_dataset": "mahmudulhaqueshawon/cat-image",
        "raw_dir": "data/raw_data",
        "split_dir": "data/data_splits",
    },
    "models": [
        "simclr_num_residual_layers1_mlp_hidden_dim512_mask_size_ratio00_tau005.pt",
        "simclr_num_residual_layers1_mlp_hidden_dim512_mask_size_ratio035_tau005.pt",
        "simclr_num_residual_layers1_mlp_hidden_dim512_mask_size_ratio00_tau0001.pt",
        "simclr_num_residual_layers1_mlp_hidden_dim512_mask_size_ratio035_tau0001.pt",
    ],
}

CONFIG["vis_dir"] = CONFIG["save_dir"] / "metrics"
CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["vis_dir"].mkdir(parents=True, exist_ok=True)


def extract_simclr_encoders() -> dict:
    """
    Loads SimCLR model encoders from specified model paths in CONFIG["models"].
    Returns a list of encoder models with loaded weights.
    """
    encoders = {}
    for model_path in CONFIG["models"]:
        model_path = Path(SSL_PREFIX) / model_path
        if not model_path.exists():
            print(f"Model {model_path} does not exist.")
            continue

        num_residual_layers = int(model_path.stem.split("_")[3][-1])
        mask = float(model_path.stem.split("_")[-2][5:])
        mlp_hidden_dim = int(model_path.stem.split("_")[6][3:])
        tau = float('.'+model_path.stem.split("_")[-1][3:])*10


        simclr_model = SimCLR(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            residual_hiddens=CONFIG["residual_hiddens"],
            num_residual_layers=num_residual_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        state_dict = torch.load(model_path, map_location=CONFIG["device"])
        simclr_model.load_state_dict(state_dict)

        encoder = simclr_model.encoder
        encoder.to(CONFIG["device"])
        encoder.eval()

        encoders[
            "simclr_vae_layers"
            + "_masked"
            + ("True" if mask > 0 else "False")
            + "_tau"
            + str(tau)
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

    encoders = extract_simclr_encoders()

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
        encoder = params["model"]

        model.encoder = encoder
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
            tensorboard=True
        )

    print("Training and visualization complete.")


if __name__ == "__main__":
    main()
