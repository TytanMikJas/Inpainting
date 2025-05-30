import os
from src.scripts.etl_process.DataDownloader import DataDownloader
from src.scripts.etl_process.DataSplitter import DataSplitter
from src.scripts.etl_process.DataTransformer import DataTransformer


class ETLProcessor:
    """
    ETLProcessor orchestrates the ETL process for a Kaggle dataset.
    It downloads the dataset, splits it into train, validation, and test sets,
    and transforms the data into PyTorch DataLoaders.
    Attributes:
        kaggle_dataset (str): The Kaggle dataset identifier in the form 'owner/dataset-name'.
        raw_dir (str): Directory where raw data will be stored.
        split_dir (str): Directory where split data will be stored.
    Methods:
        process(): Executes the ETL process, returning DataLoaders for train, validation, and test sets.
    """

    def __init__(
        self,
        kaggle_dataset: str,
        raw_dir: str = "data/raw_data",
        split_dir: str = "data/splits",
    ):
        self.kaggle_dataset = kaggle_dataset
        self.raw_dir = raw_dir
        self.split_dir = split_dir

    def process(self):
        downloader = DataDownloader(self.kaggle_dataset, self.raw_dir)
        image_dir = downloader.download()

        splitter = DataSplitter(image_dir=image_dir, split_dir=self.split_dir)
        if not all(
            os.path.exists(os.path.join(self.split_dir, f"{s}.json"))
            for s in ["train", "val", "test"]
        ):
            splitter.split()

        transformer = DataTransformer()
        train_loader = transformer.get_dataloader(
            os.path.join(self.split_dir, "train.json"), shuffle=True
        )
        val_loader = transformer.get_dataloader(
            os.path.join(self.split_dir, "val.json"), shuffle=False
        )
        test_loader = transformer.get_dataloader(
            os.path.join(self.split_dir, "test.json"), shuffle=False
        )

        return train_loader, val_loader, test_loader
