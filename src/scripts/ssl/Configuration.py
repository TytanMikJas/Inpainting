import torch
from pathlib import Path

DEFAULT_CONFIG = {
    "input_dim": 3,
    "hidden_dim": 128,
    "residual_hiddens": 64,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "param_grid": {
        "num_residual_layers": [1, 2],
        "mlp_hidden_dim": [512, 1024, 2048],
        "mask_size_ratio": [0.0, 0.35],
        "num_residual_layers": [1],
        "mlp_hidden_dim": [512],
    },
    "dataset_config": {
        "kaggle_dataset": "mahmudulhaqueshawon/cat-image",
        "raw_dir": "data/raw_data",
        "split_dir": "data/data_splits",
    },
}

def for_dictionary_mkdir(dictionary):
    dictionary["vis_dir"] = dictionary["save_dir"] / "embeddings"
    dictionary["save_dir"].mkdir(parents=True, exist_ok=True)
    dictionary["vis_dir"].mkdir(parents=True, exist_ok=True)
    return dictionary

def get_byol_config():
    byol_default_config = DEFAULT_CONFIG.copy()
    byol_default_config["param_grid"]["tau"] = [0.95, 0.97, 0.99, 0.999]
    byol_default_config["save_dir"] = Path("models/ssl/byol/")
    return for_dictionary_mkdir(byol_default_config)

def get_barlow_twins_config():
    barlow_twins_default_config = DEFAULT_CONFIG.copy()
    barlow_twins_default_config["param_grid"]["lambda_"] = [0.001, 0.005, 0.01]
    barlow_twins_default_config["save_dir"] = Path("models/ssl/barlow_twins/")
    return for_dictionary_mkdir(barlow_twins_default_config)

def get_simclr_config():
    simclr_default_config = DEFAULT_CONFIG.copy()
    simclr_default_config["param_grid"]["tau"] = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    simclr_default_config["save_dir"] = Path("models/ssl/simclr/")
    return for_dictionary_mkdir(simclr_default_config)
