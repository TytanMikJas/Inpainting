from typing import Callable, Dict, List, Tuple
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os


class Trainer:
    """
    A class to handle training of Autoencoders and VAEs.
    """

    def _apply_random_mask(
        self, images: torch.Tensor, mask_ratio: float
    ) -> torch.Tensor:
        """
        Apply a random square mask with pixel value 0.5 to each image in the batch.
        - Random size sampled from [0.15, mask_size_ratio]
        - Random location per image
        Vectorized version — no for loop.
        """
        if mask_ratio <= 0.0:
            return images

        B, _, H, W = images.shape
        device = images.device

        random_ratios = torch.empty(B, device=device).uniform_(0.15, mask_ratio)
        mask_sizes = (random_ratios * H).long().clamp(min=1)

        top_coords = torch.randint(0, H, (B,), device=device)
        left_coords = torch.randint(0, W, (B,), device=device)

        top_coords = torch.minimum(top_coords, H - mask_sizes)
        left_coords = torch.minimum(left_coords, W - mask_sizes)

        for i in range(B):
            y1, y2 = top_coords[i], top_coords[i] + mask_sizes[i]
            x1, x2 = left_coords[i], left_coords[i] + mask_sizes[i]
            images[i, :, y1:y2, x1:x2] = 0.5

        return images

    def _initialize_metrics(
        self,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        train_metrics = {
            "loss": [],
            "mse": [],
            "partial_loss": [],
            "step": [],
            "num_active_dims": [],
        }
        val_metrics = {
            "loss": [],
            "mse": [],
            "partial_loss": [],
            "step": [],
            "num_active_dims": [],
        }
        return train_metrics, val_metrics

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        train_metrics: Dict[str, List[float]],
        mask_ratio: float,
    ) -> None:
        model.train()
        total_loss = total_mse = total_partial_loss = total_num_active_dims = 0.0

        for x_batch, _ in tqdm(train_loader, desc="Training"):
            x_batch = x_batch.to(model.device)
            x_masked = x_batch.clone()
            x_masked = self._apply_random_mask(x_masked, mask_ratio)
            optimizer.zero_grad()

            out = model(x_masked)
            if isinstance(out, dict):
                recon = out["recon"]
                partial_loss = out.get(
                    "partial_loss", torch.tensor(0.0, device=x_masked.device)
                )
                num_active_dims = out.get("num_active_dims", 0)
            else:
                recon = out
                partial_loss = torch.tensor(0.0, device=x_masked.device)

            recon_loss = loss_fn(recon, x_batch)
            loss = recon_loss + partial_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss.item()
                total_mse += nn.functional.mse_loss(
                    recon.view(x_batch.size(0), -1),
                    x_batch.view(x_batch.size(0), -1),
                    reduction="mean",
                ).item()
                total_partial_loss += (
                    partial_loss.item()
                    if isinstance(partial_loss, torch.Tensor)
                    else float(partial_loss)
                )
                total_num_active_dims += num_active_dims

        loss = total_loss / len(train_loader)
        mse = total_mse / len(train_loader)
        partial_loss = total_partial_loss / len(train_loader)
        num_active_dims = total_num_active_dims / len(train_loader)

        print(
            f"Train - Loss: {loss:.6f}, MSE: {mse:.6f}, Partial Loss: {partial_loss:.6f}, "
            f"Num Active Dims: {num_active_dims:.2f}"
        )

        train_metrics["loss"].append(loss)
        train_metrics["mse"].append(
            mse.item() if isinstance(mse, torch.Tensor) else float(mse)
        )
        train_metrics["partial_loss"].append(
            partial_loss.item()
            if isinstance(partial_loss, torch.Tensor)
            else float(partial_loss)
        )
        train_metrics["num_active_dims"].append(num_active_dims)

    def _validate_model(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: Callable,
        val_metrics: Dict[str, List[float]],
    ) -> None:
        model.eval()
        total_loss = total_mse = total_partial_loss = total_num_active_dims = 0.0
        batches = 0

        with torch.no_grad():
            for masked, clean in val_loader:
                masked = masked.to(model.device)
                clean = clean.to(model.device)

                out = model(masked)
                if isinstance(out, dict):
                    recon = out["recon"]
                    partial_loss = out.get(
                        "partial_loss", torch.tensor(0.0, device=masked.device)
                    )
                    num_active_dims = out.get("num_active_dims", 0)
                else:
                    recon = out
                    partial_loss = torch.tensor(0.0, device=masked.device)

                recon_loss = loss_fn(recon, clean)
                mse = nn.functional.mse_loss(
                    recon.view(clean.size(0), -1),
                    clean.view(clean.size(0), -1),
                    reduction="mean",
                )

                total_loss += (recon_loss + partial_loss).item()
                total_mse += mse.item()
                total_partial_loss += partial_loss.item()
                total_num_active_dims += num_active_dims
                batches += 1

        loss = total_loss / batches
        mse = total_mse / batches
        partial_loss = total_partial_loss / batches
        total_num_active_dims = total_num_active_dims / batches
        val_metrics["loss"].append(loss)
        val_metrics["mse"].append(mse)
        val_metrics["partial_loss"].append(partial_loss)
        val_metrics["num_active_dims"].append(total_num_active_dims)
        print(
            f"Validation - Loss: {loss:.6f}, MSE: {mse:.6f}, partial_loss: {partial_loss:.6f}, num_active_dims: {total_num_active_dims}"
        )

        return loss

    def plot_metrics(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]],
        vis_dir: str,
        model_name: str,
    ):
        fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=True)
        ax1, ax2, ax3, ax4 = axes

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

        ax3.plot(
            train_metrics["step"],
            train_metrics["partial_loss"],
            label="train partial loss",
        )
        ax3.plot(
            val_metrics["step"], val_metrics["partial_loss"], label="val partial loss"
        )
        ax3.set_ylabel("Partial Loss")
        ax3.set_xlabel("Step")
        ax3.grid()
        ax3.legend()

        ax4.plot(
            train_metrics["step"],
            train_metrics["num_active_dims"],
            label="train Perplexity",
        )
        ax4.plot(
            val_metrics["step"],
            val_metrics["num_active_dims"],
            label="val Perplexity",
        )
        ax4.set_ylabel("Perplexity")
        ax4.set_xlabel("Step")
        ax4.grid()
        ax4.legend()
        
        fig.suptitle(f"{model_name}", fontsize=16)

        fig.tight_layout()

        save_path = os.path.join(vis_dir, f"{model_name}_metrics.png")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(save_path)
        plt.close(fig)

    def train_supervised(
        self,
        model: nn.Module,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float,
        weight_decay: float,
        mask_ratio: float,
        loss_fn: Callable,
        model_name: str,
        save_dir: str,
        vis_dir: str,
    ) -> None:
        train_metrics, val_metrics = self._initialize_metrics()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self._train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                train_metrics,
                mask_ratio,
            )
            loss = self._validate_model(model, val_loader, loss_fn, val_metrics)
            train_metrics["step"].append(epoch)
            val_metrics["step"].append(epoch)
            if loss < best_val_loss:
                best_val_loss = loss
                torch.save(model.state_dict(), f"{save_dir}/{model_name}.pt")

        self.plot_metrics(train_metrics, val_metrics, vis_dir, model_name)
