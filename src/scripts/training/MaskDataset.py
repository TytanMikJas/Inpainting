from torch.utils.data import Dataset
import random


class MaskedDataset(Dataset):
    def __init__(self, clean_dataset: Dataset, mask_size_ratio: float):
        """
        :param clean_dataset: Dataset providing clean images.
        :param mask_size_ratio: Size of square mask as a fraction of image size (e.g., 0.3 masks a 30% wide/tall square).
        """
        self.clean = clean_dataset
        self.mask_size_ratio = mask_size_ratio

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean_img, _ = self.clean[idx]
        masked_img = clean_img.clone()

        _, H, W = clean_img.shape
        mask_size = int(min(H, W) * self.mask_size_ratio)

        top = random.randint(0, H - mask_size)
        left = random.randint(0, W - mask_size)

        masked_img[:, top : top + mask_size, left : left + mask_size] = 0

        return masked_img, clean_img
