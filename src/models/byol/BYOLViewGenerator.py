from torchvision.transforms import v2
import torch
from typing import Tuple


class BYOLViewGenerator:
    """
    BYOLViewGenerator applies a series of transformations to the input images to create two different views,
    which are then used for training the BYOL model.
    Args:
        device (str): Device to run the transformations on ('cpu' or 'cuda').
        image_size (int): Size of the input images (default is 64).
    """

    def __init__(self, device: str = "cpu", image_size: int = 64):
        self.device = device
        self.transform = self._get_byol_augmentation(
            image_size, gaussian_blur_prob=1.0
        ).to(self.device)
        self.transform_prim = self._get_byol_augmentation(
            image_size, gaussian_blur_prob=0.1
        ).to(self.device)

    def _get_byol_augmentation(
        self, image_size: int, gaussian_blur_prob: float
    ) -> v2.Compose:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                    p=gaussian_blur_prob,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __call__(self, x_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.transform(x_batch)
        v_prim = self.transform_prim(x_batch)
        print(
            f"v min/max: {v.min().item():.3f}/{v.max().item():.3f}, "
            f"v_prim min/max: {v_prim.min().item():.3f}/{v_prim.max().item():.3f}"
        )
        return v.to(self.device), v_prim.to(self.device)
