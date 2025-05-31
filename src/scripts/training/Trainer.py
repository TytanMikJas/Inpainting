from typing import Callable, Dict, List, Tuple
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    """
    A class to handle training of Autoencoders and VAEs.
    """

    def _initialize_metrics(
        self,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        train_metrics = {"loss": [], "mse": [], "kl": [], "step": []}
        val_metrics = {"loss": [], "mse": [], "kl": [], "step": []}
        return train_metrics, val_metrics

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        train_metrics: Dict[str, List[float]],
        global_step: int,
    ) -> int:
        model.train()
        pbar = tqdm(train_loader, desc="train step", leave=False)

        for noisy, clean in pbar:
            noisy = noisy.to(model.device)
            clean = clean.to(model.device)
            optimizer.zero_grad()

            out = model(noisy)
            if isinstance(out, dict):
                recon = out["recon"]
                kl = out.get("partial_loss", torch.tensor(0.0, device=noisy.device))
            else:
                recon = out
                kl = torch.tensor(0.0, device=noisy.device)

            recon_loss = loss_fn(recon, clean)
            loss = recon_loss + kl
            loss.backward()
            optimizer.step()

            batch_size = clean.size(0)
            train_metrics["loss"].append(loss.item())
            train_metrics["mse"].append(
                nn.functional.mse_loss(
                    recon.view(batch_size, -1),
                    clean.view(batch_size, -1),
                    reduction="mean",
                ).item()
            )
            train_metrics["kl"].append(
                kl.item() if isinstance(kl, torch.Tensor) else float(kl)
            )
            train_metrics["step"].append(global_step)

            global_step += 1
            pbar.update(1)

        pbar.close()
        return global_step

    def _validate_model(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: Callable,
        val_metrics: Dict[str, List[float]],
        global_step: int,
    ) -> None:
        model.eval()
        total_loss = total_mse = total_kl = 0.0
        batches = 0

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(model.device)
                clean = clean.to(model.device)

                out = model(noisy)
                if isinstance(out, dict):
                    recon = out["recon"]
                    kl = out.get("partial_loss", torch.tensor(0.0, device=noisy.device))
                else:
                    recon = out
                    kl = torch.tensor(0.0, device=noisy.device)

                recon_loss = loss_fn(recon, clean)
                mse = nn.functional.mse_loss(
                    recon.view(clean.size(0), -1),
                    clean.view(clean.size(0), -1),
                    reduction="mean",
                )

                total_loss += (recon_loss + kl).item()
                total_mse += mse.item()
                total_kl += kl.item()
                batches += 1

        if batches > 0:
            loss = total_loss / batches
            mse = total_mse / batches
            kl = total_kl / batches
            val_metrics["loss"].append(loss)
            val_metrics["mse"].append(mse)
            val_metrics["kl"].append(kl)
            val_metrics["step"].append(global_step)
            print(f"Validation - Loss: {loss:.4f}, MSE: {mse:.4f}, KL: {kl:.4f}")

    def plot_metrics(
        self, train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]]
    ):
        fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)
        ax1, ax2, ax3 = axes

        ax1.plot(train_metrics["step"], train_metrics["loss"], label="train loss")
        ax1.plot(val_metrics["step"], val_metrics["loss"], label="val loss")
        ax1.set_ylabel("Loss")
        ax1.grid()
        ax1.legend()

        ax2.plot(train_metrics["step"], train_metrics["mse"], label="train mse")
        ax2.plot(val_metrics["step"], val_metrics["mse"], label="val mse")
        ax2.set_ylabel("MSE")
        ax2.grid()
        ax2.legend()

        ax3.plot(train_metrics["step"], train_metrics["kl"], label="train kl")
        ax3.plot(val_metrics["step"], val_metrics["kl"], label="val kl")
        ax3.set_ylabel("KL")
        ax3.set_xlabel("Step")
        ax3.grid()
        ax3.legend()

        plt.show()

    def train_supervised(
        self,
        model: nn.Module,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float,
        loss_fn: Callable,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        train_metrics, val_metrics = self._initialize_metrics()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        global_step = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            global_step = self._train_epoch(
                model, train_loader, optimizer, loss_fn, train_metrics, global_step
            )
            self._validate_model(model, val_loader, loss_fn, val_metrics, global_step)

        return train_metrics, val_metrics
