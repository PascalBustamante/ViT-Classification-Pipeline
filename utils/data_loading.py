import torch
import os
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm
from PIL import Image
from enum import Enum, auto

from utils.data_exploration import calculate_mean_std, calculate_label_distribution
from utils.logger import Logger

log_file_path = "logs/data_loading_log.txt"
logger = Logger(__name__, log_file_path)


class DataLoaderManager:
    """
    DataLoaderManager class for managing data loading and processing.

    Args:
        batch_size (int): Batch size for data loading.
        train_files (list): List of training data files.
        val_files (list): List of validation data files.
        test_files (list): List of test data files.
        id_to_breed (dict): Mapping of image IDs to dog breeds.
        num_workers (int): Number of CPU cores dedicated to data loading.
    """

    def __init__(
        self, batch_size, train_files, val_files, test_files, id_to_breed, num_workers
    ):
        self.batch_size = batch_size
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.id_to_breed = id_to_breed
        self.num_workers = 0
        self.mean = 0 
        self.std = 0
        self.label_distribution = {}
        self.calculate_statistics(
            image_files=self.train_files, id_to_breed=self.id_to_breed
        )

        # Log the number of workers used for data loading
        logger.info(f"Number of workers for data loading: {self.num_workers}")
        # Log training dataset statistics
        logger.info(f"Dataset Statistics for training data:")
        logger.info(f" - Mean: {self.mean}")
        logger.info(f" - Standard Deviation: {self.std}")
        logger.info(f" - Label Distribution: {self.label_distribution}")

    # functions from utils
    def calculate_statistics(self, image_files, id_to_breed):
        self.mean, self.std = calculate_mean_std(image_files) 
        self.label_distribution = calculate_label_distribution(
            id_to_breed, image_files
        )

    def get_dataloader(self, dataset_type):
        """
        Create a data loader for a specific dataset type (train, val, or test).

        Args:
            dataset_type (str): Type of dataset (train, val, or test).

        Returns:
            DataLoader: DataLoader for the specified dataset type.
        """
        logger.info(f"Creating data loader for dataset type: {dataset_type}")
        '''loading_time = timeit.timeit(
            stmt=lambda: self.preload(dataset_type),  # Function that loads data
            number=1,  # Execute the function once
        )'''

        shuffle = True
        if dataset_type == "train":
            transform = transforms.Compose(
                [
                    #transforms.ToPILImage(),
                    transforms.RandomRotation(15),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            dataset = DOGGODataset(
                self.train_files,
                self.id_to_breed,
                dataset_type="train",
                transform=transform,
            )

        elif dataset_type in ["val", "test"]:
            transform = transforms.Compose(
                [
                    #transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            if dataset_type == "val":
                dataset = DOGGODataset(
                    self.val_files,
                    self.id_to_breed,
                    dataset_type="val",
                    transform=transform,
                )
            else:
                shuffle = False
                dataset = DOGGODataset(
                    self.test_files,
                    self.id_to_breed,
                    dataset_type="test",
                    transform=transform,
                )

        # Log the number of samples for the current dataset type
        logger.info(f"Number of {dataset_type} samples: {len(dataset)}")

        # Log dataset information (e.g., number of classes)
        logger.info(f"Number of classes: {len(set(self.id_to_breed.values()))}")

        # Log data preprocessing details (if applicable)
        if dataset_type in ["train", "val", "test"]: 
            logger.info("Data preprocessing details for training data:")
            logger.info(
                " - Data augmentation: RandomRotation(15)"
            )  # make it a variable or change it

        else:
            logger.error(
                f"ValueError:{dataset_type} is not of type 'train', 'val', or 'test'"
            )
            raise ValueError("dataset_type should be 'train', 'val', or 'test'")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        # Log the data loading time
       # logger.info(f"Data loading time for {dataset_type} data: {loading_time:.2f}s")

         # Wrap the DataLoader with DataPrefetcher
        #prefetcher = DataPrefetcher(dataloader)
        #prefetcher.preload()

        # Log the data loading time
        #logger.info(f"Data loading time for {dataset_type} data: {loading_time:.2f}s")

        return dataloader 


class DOGGODataset(Dataset):
    def __init__(self, image_files, id_to_breed, dataset_type, transform=None):
        super().__init__()
        self.image_files = image_files
        self.id_to_breed = id_to_breed
        self.dataset_type = dataset_type
        self.transform = transform

        # Create Enum of breeds
        self.breeds_enum = create_dog_breed_enum(set(id_to_breed.values()))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_id = os.path.splitext(os.path.basename(image_file))[0]

        image = Image.open(image_file)
        if self.transform:
            image = self.transform(image)

        if (
            self.dataset_type == "test"
        ):  # for later differences, like maybe not returning labels for future test
            breed = self.id_to_breed[image_id]
            label = self.breeds_enum[breed.upper()].value
            label = torch.tensor(
                label, dtype=torch.long
            )  # Convert the label to a tensor
            return {"image": image, "label": label}  # turn it into a dict
        else:
            breed = self.id_to_breed[image_id]
            label = self.breeds_enum[breed.upper()].value
            label = torch.tensor(
                label, dtype=torch.long
            )  # Convert the label to a tensor
            return {"image": image, "label": label}  # turn it into a dict


class DataPrefetcher:   ### Still not implemented!!!!!
    # DataPrefetcher class for prefetching data using CUDA streams.
    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.next_data = None

    def preload(self):
        try:
            self.next_data = next(self.iterator)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            for key, value in self.next_data.items():
                if isinstance(value, torch.Tensor):
                    self.next_data[key] = value.cuda(non_blocking=True)
            self.next_data["image"] = self.next_data["image"].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


def create_dog_breed_enum(breeds):
    return Enum(
        "Dog Breed", {breed.upper(): i for i, breed in enumerate(breeds, start=1)}
    )
