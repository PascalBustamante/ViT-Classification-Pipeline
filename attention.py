import torch
from torch import nn

from config.config import Config

class AttentionHead(nn.Model):
    def __init__(self, 
            model_dim, 
            head_size=Config.HEAD_SIZE): 
        super().__init__()

        self.head_size = head_size

        # Linear layers to compute the attention logits
        self.query = nn.Linear(model_dim, head_size)
        self.key = nn.Linear(model_dim, head_size)
        self.value = nn.Linear(model_dim, head_size)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2,-1)

        # Scaling
        attention = attention / (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)

        attention = attention @ V

        return attention
    
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.head_size = d_model // n_heads

    self.W_o = nn.Linear(d_model, d_model)

    self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

  def forward(self, x):
    # Combine attention heads
    out = torch.cat([head(x) for head in self.heads], dim=-1)

    out = self.W_o(out)

    return out

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    from torchvision import transforms
    from torch.utils.data import DataLoader


    from preprocessing.dataset import ImageClassificationDataset\
    
    attention_head = AttentionHead()

    # Create a sample DataFrame with labels
    labels_df = pd.read_csv(Config.LABELS_PATH)

    samples = labels_df.sample(n=3)
    image_files = []

    for _, row in samples.iterrows():
        image_file = os.path.join(Config.TRAIN_DIR, f"{row['id']}.jpg")
        image_files.append(image_file)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    dataset = ImageClassificationDataset(image_files, dataset_type="train", transform=transform, labels_df=labels_df)
    
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)

    
    for batch in dataloader:
        #print(type(batch))
        imgs = batch[0]
        labels = batch[1]

        out = patch_embedder.forward(imgs[0])
        print(out.shape)
