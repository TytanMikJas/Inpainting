from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
from torch.utils.data import DataLoader
import torch
import os
import json
from torch.utils.data import RandomSampler
from src.training.Trainer import Trainer
from src.training.MaskDataset import MaskedDataset


@dataclass
class SingleTrainEvent:
    """
    A single training event containing all necessary parameters for training a model.
    """

    epochs: int
    train_loader: DataLoader
    val_loader: DataLoader
    lr: float
    loss_fn: Callable
    loss_fn_args: Optional[Tuple[Any]]
    mask_size: float = 0.0
    save_dir: Optional[str] = None


class TrainScheduler:
    """
    TrainScheduler orchestrates the training process for a model using multiple training events.
    It manages the training loop, including epochs, data loaders, and loss functions.
    Attributes:
        train_events (List[SingleTrainEvent]): List of training events to be executed.
        trainer (Trainer): Instance of the Trainer class to handle training logic.
        model (Any): The model to be trained.
    """

    def __init__(
        self,
        SingleTrainEvents: List[SingleTrainEvent],
        model: Any,
        model_name: str = "Model",
    ):
        self.train_events = SingleTrainEvents
        self.trainer: Trainer = Trainer()
        self.model = model
        self.model_name = model_name

    def start_training(self):
        for i, event in enumerate(self.train_events):
            print(f"Start {i} training event with {event.epochs} epochs.")
            self.run_event(event, index=i)
        print("All training events completed.")

    def prepare_event(self, event: SingleTrainEvent):
        print(
            f"[INFO] Preparing event: epochs={event.epochs}, lr={event.lr}, mask_ratio={event.mask_size}"
        )

        masked_train_ds = MaskedDataset(event.train_loader.dataset, event.mask_size)
        masked_val_ds = MaskedDataset(event.val_loader.dataset, event.mask_size)

        def clone_loader(orig_loader, new_ds):
            sampler = orig_loader.sampler
            has_shuffle = isinstance(sampler, RandomSampler)

            return DataLoader(
                new_ds,
                batch_size=orig_loader.batch_size,
                shuffle=has_shuffle,
                num_workers=orig_loader.num_workers,
                pin_memory=orig_loader.pin_memory,
                drop_last=orig_loader.drop_last,
            )

        self.train_loader = clone_loader(event.train_loader, masked_train_ds)
        self.val_loader = clone_loader(event.val_loader, masked_val_ds)

    def run_event(self, event: SingleTrainEvent, index: int):
        """Prepare data, train model, and save results."""
        self.prepare_event(event)

        save_dir = event.save_dir or os.path.join("data/training/", self.model_name)
        os.makedirs(save_dir, exist_ok=True)

        train_metrics, val_metrics = self.trainer.train_supervised(
            model=self.model,
            epochs=event.epochs,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            lr=event.lr,
            loss_fn=event.loss_fn,
        )

        model_path = os.path.join(save_dir, f"{self.model_name}_event{index}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path} (event {index})")

        metrics = {"train": train_metrics, "val": val_metrics}
        metrics_path = os.path.join(
            save_dir, f"{self.model_name}_event{index}_metrics.json"
        )
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"Metrics saved to {metrics_path}")
