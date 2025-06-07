import sys
import itertools
from pathlib import Path
import torch
from torch import optim

from src.scripts.etl_process.ETLProcessor import ETLProcessor
from src.scripts.ssl.SSLTrainer import SSLTrainer
from src.models.ssl.byol.BYOL import BYOL
from src.models.ssl.barlow_twins.BarlowTwins import BarlowTwins
from src.scripts.ssl.Configuration import get_byol_config, get_barlow_twins_config


MODEL_REGISTRY = {
    "byol": {
        "model_cls": BYOL,
        "config_fn": get_byol_config,
        "loss_fn": lambda model, q, z: model.byol_loss(q, z),
        "update_fn": lambda model: model.update_target_network(),
    },
    "barlow_twins": {
        "model_cls": BarlowTwins,
        "config_fn": get_barlow_twins_config,
        "loss_fn": lambda model, z1, z2: model.barlow_twins_loss(z1, z2),
        "update_fn": lambda model: None,  
    }
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODEL_REGISTRY:
        print("Usage: python run_ssl.py [byol|barlow_twins]")
        sys.exit(1)

    model_key = sys.argv[1]
    config = MODEL_REGISTRY[model_key]["config_fn"]()
    model_cls = MODEL_REGISTRY[model_key]["model_cls"]
    loss_fn = MODEL_REGISTRY[model_key]["loss_fn"]
    update_fn = MODEL_REGISTRY[model_key]["update_fn"]

    print("Preparing data...")
    etl = ETLProcessor(**config["dataset_config"])
    train_loader, _, _ = etl.process()

    param_keys = list(config["param_grid"].keys())
    param_values = list(config["param_grid"].values())
    param_combinations = list(itertools.product(*param_values))

    total_configs = len(param_combinations)
    print(f"Total configurations to run: {total_configs}")

    for i, combo in enumerate(param_combinations):
        model_args = {
            "input_dim": config["input_dim"],
            "hidden_dim": config["hidden_dim"],
            "residual_hiddens": config["residual_hiddens"],
            "device": config["device"],
        }
        model_args.update(dict(zip(param_keys, combo)))

        model_name = f"{model_key}_" + "_".join(f"{k}{str(v).replace('.', '')}" for k, v in zip(param_keys, combo))

        print(f"\n[{i + 1}/{total_configs}] Running: {model_name}")

        trainer = SSLTrainer(
            model_cls=model_cls,
            model_args=model_args,
            device=config["device"],
            save_dir=config["save_dir"],
        )

        trained_model, final_loss = trainer.train(
            dataloader=train_loader,
            epochs=config["epochs"],
            model_name=model_name,
            loss_fn=loss_fn,
            update_fn=update_fn,
        )

        if final_loss is not None:
            trainer.visualize(
                model=trained_model,
                dataloader=train_loader,
                model_name=model_name,
                final_loss=final_loss,
            )

    print("Training and visualization complete.")


if __name__ == "__main__":
    main()
