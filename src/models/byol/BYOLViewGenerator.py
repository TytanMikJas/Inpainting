from torchvision.transforms import v2
import torch
import random
from typing import Tuple


class BYOLViewGenerator:
    """
    BYOLViewGenerator applies a series of transformations to the input images to create two different views,
    which are then used for training the BYOL model.
    Args:
        device (str): Device to run the transformations on ('cpu' or 'cuda').
        image_size (int): Size of the input images (default is 64).
        masked_aug (float): Whether to apply masked augmentation, and how big.
    """

    def __init__(
        self, device: str = "cpu", image_size: int = 64, mask_size_ratio: float = 0.0
    ):
        self.device = device
        self.mask_size_ratio = mask_size_ratio
        self.transform = self._get_byol_augmentation(
            image_size, gaussian_blur_prob=1.0
        ).to(self.device)
        self.transform_prim = self._get_byol_augmentation(
            image_size, gaussian_blur_prob=0.1
        ).to(self.device)

    def _apply_random_mask(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply a random square mask with pixel value 0.5 to each image in the batch.
        - Random size sampled from [0.15, mask_size_ratio]
        - Random location per image
        Vectorized version — no for loop.
        """
        if self.mask_size_ratio <= 0.0:
            return images

        B, _, H, W = images.shape
        device = images.device

        random_ratios = torch.empty(B, device=device).uniform_(
            0.15, self.mask_size_ratio
        )
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

    def _get_byol_augmentation(
        self, image_size: int, gaussian_blur_prob: float
    ) -> v2.Compose:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=5, sigma=(0.4, 2))],
                    p=gaussian_blur_prob,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __call__(self, x_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.transform(x_batch)
        v_prim = self.transform_prim(x_batch)

        v = torch.nan_to_num(v, nan=0.0)
        v_prim = torch.nan_to_num(v_prim, nan=0.0)

        if self.mask_size_ratio > 0.0:
            if random.random() < 0.5:
                v = self._apply_random_mask(v)
            else:
                v_prim = self._apply_random_mask(v_prim)

        return v.to(self.device), v_prim.to(self.device)
