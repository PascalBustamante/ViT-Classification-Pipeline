import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt


from config.config import Config

class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels,
                image_size=Config.IMG_SIZE,
                patch_size=Config.PATCH_SIZE, 
                in_channels=Config.IN_CHANNELS, 
                embed_dim=Config.EMBED_DIM,
                dropout=Config.DROPOUT):
    super().__init__()

    self.d_model = d_model # Dimensionality of Model
    self.img_size = img_size # Image Size
    self.patch_size = patch_size # Patch Size
    self.n_channels = n_channels # Number of Channels

    self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

  # B: Batch Size
  # C: Image Channels
  # H: Image Height
  # W: Image Width
  # P_col: Patch Column
  # P_row: Patch Row
  def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

    x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

    x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
    
    return x
  


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    from torchvision import transforms
    from torch.utils.data import DataLoader


    from preprocessing.dataset import ImageClassificationDataset

    # Instantiate PatchEmbedding module
    patch_embedder = PatchEmbedding()

    # Create a sample DataFrame with labels
    labels_df = pd.read_csv(Config.LABELS_PATH)

    samples = labels_df.sample(n=3)
    image_files = []

    for _, row in samples.iterrows():
        image_file = os.path.join(Config.TRAIN_DIR, f"{row['id']}.jpg")
        image_files.append(image_file)
        #print(image_file)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    dataset = ImageClassificationDataset(image_files, dataset_type="train", transform=transform, labels_df=labels_df)
    
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)

    #fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a row of 3 subplots

    for batch in dataloader:
        #print(type(batch))
        imgs = batch[0]
        labels = batch[1]

        out = patch_embedder.forward(imgs)
        print(out.shape)