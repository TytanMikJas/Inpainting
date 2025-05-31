import torch
from torch.utils.data import Dataset


class NoisyDataset(Dataset):
    def __init__(self, clean_dataset: Dataset, noise_level: float):
        self.clean = clean_dataset
        self.noise_level = noise_level

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean_img, _ = self.clean[idx]

        noise = torch.randn_like(clean_img) * self.noise_level
        noisy_img = clean_img + noise
        noisy_img = noisy_img.clamp(0.0, 1.0)
        return noisy_img, clean_img
