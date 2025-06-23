from pathlib import Path
from typing import Callable, Optional, Type
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class SSLTrainer:
    def __init__(
        self,
        model_cls: Type[nn.Module],
        model_args: dict,
        device: str,
        save_dir: Path,
        vis_dir: Optional[Path] = None,
    ):
        """
        Args:
            model_cls (nn.Module): Class of the SSL model (e.g. BYOL or BarlowTwins).
            model_args (dict): Arguments passed to the model constructor.
            device (str): Device to use ('cuda' or 'cpu').
            save_dir (Path): Where to save model checkpoints.
            vis_dir (Path, optional): Where to save visualizations.
        """
        self.model_cls = model_cls
        self.model_args = model_args
        self.device = device
        self.save_dir = save_dir
        self.vis_dir = vis_dir or save_dir / "embeddings"

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        model_name: str,
        loss_fn: Callable,
        update_fn: Optional[Callable] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> tuple[nn.Module, float]:
        """
        Train the SSL model.

        Returns:
            Trained model, final average loss.
        """
        model = self.model_cls(**self.model_args).to(self.device)
        optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for x_batch, _ in tqdm(dataloader, desc=f"[{model_name}] Epoch {epoch}"):
                x_batch = x_batch.to(self.device)
                out1, out2 = model(x_batch)
                loss = loss_fn(model, out1, out2)

                if not torch.isfinite(loss).all():
                    print("Non-finite loss encountered.")
                    return model, None

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                if update_fn is not None:
                    update_fn(model)

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.8f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), self.save_dir / f"{model_name}.pt")

        return model, best_loss

    def visualize(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str,
        final_loss: float,
        max_samples: int = 1024,
    ):
        """
        Visualize embeddings using t-SNE.
        """
        model.eval()
        all_embeddings = []
        total = 0

        with torch.no_grad():
            for x_batch, _ in dataloader:
                x_batch = x_batch.to(self.device)
                feats = model.forward_repr(x_batch)
                feats = torch.flatten(feats, 1)
                all_embeddings.append(feats.cpu())
                total += feats.size(0)
                if total >= max_samples:
                    break

        embeddings = torch.cat(all_embeddings)[:max_samples]
        reduced = TSNE(
            n_components=2, perplexity=100, max_iter=1000, random_state=42
        ).fit_transform(embeddings)

        plt.figure(figsize=(8, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7, c="navy")
        plt.title(f"{model_name}\nFinal loss: {final_loss:.6f}")
        plt.tight_layout()
        plt.savefig(self.vis_dir / f"{model_name}_tsne.png")
        plt.close()
