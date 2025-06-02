import os
from pathlib import Path
import itertools
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.models.byol.BYOL import BYOL
from src.scripts.etl_process.ETLProcessor import ETLProcessor


CONFIG = {
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
        "tau": [0.95, 0.97, 0.99, 0.999],
    },
    "save_dir": Path("models/ssl/byol/"),
    "dataset_config": {
        "kaggle_dataset": "mahmudulhaqueshawon/cat-image",
        "raw_dir": "data/raw_data",
        "split_dir": "data/data_splits",
    },
}

CONFIG["vis_dir"] = CONFIG["save_dir"] / "embeddings"
CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["vis_dir"].mkdir(parents=True, exist_ok=True)


# Training + Eval
def fit_byol(
    model: BYOL,
    optimiser: optim.Optimizer,
    dataloader: DataLoader,
    epochs: int,
    device: str,
    model_name: str,
) -> float | None:
    """
    Train a BYOL model with the given dataloader and optimizer.

    Args:
        model (BYOL): The BYOL model instance.
        optimiser (Optimizer): Optimizer for training.
        dataloader (DataLoader): Training data loader.
        epochs (int): Number of training epochs.
        device (str): Device string.
        model_name (str): Name used for saving the model.

    Returns:
        float | None: Final average loss, or None if training was interrupted.
    """
    model.to(device)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, _ in tqdm(dataloader, desc=f"[{model_name}] Epoch {epoch}"):
            x_batch = x_batch.to(device)
            q, z = model(x_batch)
            loss = model.byol_loss(q, z)

            if not torch.isfinite(loss).all():
                print("Non-finite loss encountered, stopping training.")
                return None

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            model.update_target_network()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: BYOL Loss = {avg_loss:.8f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(), os.path.join(CONFIG["save_dir"], f"{model_name}.pt")
            )

    return best_loss


def visualize_embeddings(
    model: BYOL,
    dataloader: DataLoader,
    model_name: str,
    final_loss: float,
    max_samples=1024,
) -> None:
    """
    Visualizes t-SNE reduced embeddings from BYOL model's encoder.

    Args:
        model (BYOL): Trained BYOL model.
        dataloader (DataLoader): Data loader for embedding extraction.
        model_name (str): Model configuration identifier.
        final_loss (float): Final loss value to display.
        max_samples (int): Max number of samples for t-SNE.
    """
    model.eval()
    all_embeddings = []
    total = 0

    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(model.device)
            feats = model.online_encoder(x_batch)
            feats = torch.flatten(feats, 1)
            all_embeddings.append(feats.detach().cpu())
            total += feats.size(0)
            if total >= max_samples:
                break

    embeddings = torch.cat(all_embeddings)[:max_samples]

    tsne = TSNE(n_components=2, perplexity=100, max_iter=1000, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7, c="navy")
    plt.title(f"{model_name}\nFinal loss: {final_loss:.6f}")
    plt.tight_layout()

    save_path = os.path.join(CONFIG["vis_dir"], f"{model_name}_tsne.png")
    plt.savefig(save_path)
    plt.close()


def main():
    print("Preparing data...")
    etl = ETLProcessor(**CONFIG["dataset_config"])
    train_loader, _, _ = etl.process()

    param_combinations = list(itertools.product(*CONFIG["param_grid"].values()))
    total_configs = len(param_combinations)

    print(f"Total configurations to run: {total_configs}")

    for i, (layers, mlp_dim, mask_ratio, tau) in enumerate(param_combinations):
        model_name = f"byol_l{layers}_mlp{mlp_dim}_mask{mask_ratio}_tau{tau}".replace(
            ".", ""
        )
        print(f"\n[{i + 1}/{total_configs}] Running: {model_name}")

        byol = BYOL(
            input_dim=CONFIG["input_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            num_residual_layers=layers,
            residual_hiddens=CONFIG["residual_hiddens"],
            mlp_hidden_dim=mlp_dim,
            mask_size_ratio=mask_ratio,
            tau=tau,
            device=CONFIG["device"],
        )

        optimiser = optim.Adam(
            byol.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
        )

        final_loss = fit_byol(
            model=byol,
            optimiser=optimiser,
            dataloader=train_loader,
            epochs=CONFIG["epochs"],
            device=CONFIG["device"],
            model_name=model_name,
        )

        if final_loss is not None:
            visualize_embeddings(byol, train_loader, model_name, final_loss)

    print("Training and visualization complete.")


if __name__ == "__main__":
    main()
