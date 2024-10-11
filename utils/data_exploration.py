import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def calculate_mean_std(image_files):
    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Calculate the mean and standard deviation
    mean = 0.0
    std = 0.0
    for image_file in image_files:
        image = Image.open(image_file)
        image_tensor = transform(image)
        mean += image_tensor.mean()
        std += image_tensor.std()

    # Calculate the average mean and standard deviation
    mean /= len(image_files)
    std /= len(image_files)

    return mean, std
    


def calculate_label_distribution(labels_df, image_files):

    label_distribution = {}

    for image_file in image_files:
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        label = labels_df.loc[labels_df['id'] == image_id, 'breed'].values[0]

        if label is not None:
            label_distribution[label] = label_distribution.get(label, 0) + 1

    return label_distribution

def calculate_image_size_distribution(image_files):
    sizes = []

    for image_file in image_files:
        image = cv2.imread(image_file)
        print(image.shape, image.size)
        sizes.append(image.size)


if __name__ == '__main__':
    import glob

    # Load image file paths, and labels
    image_files_train_val = glob.glob(
        r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\train/*.jpg"
    )
    image_files_test = glob.glob(
        r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\test/*.jpg"
    )
    labels_df = pd.read_csv(
        r"C:\Users\pasca\CNN Doggo\dog_breed_classifier\data\DOGGO\labels.csv"
    )

    stuff = calculate_image_size_distribution(image_files_train_val)

    print(stuff)