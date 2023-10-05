import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, datasets

from tqdm import tqdm

image_info = pd.read_csv('../data/default_info.csv')

n = 0
img_name = image_info.iloc[n, 0]
id = image_info.iloc[n, 1]
component = image_info.iloc[n, 2]
label = "bad"

print('Image name: {}'.format(img_name))
print('id: {}'.format(id))
print('component: {}'.format(component))

# USING
# train_dataset = torchvision.datasets.ImageFolder(root='train')
# valid_dataset = torchvision.datasets.ImageFolder(root='valid')

# # Next, we can very easily create training and validation data loaders.
# train_loader = DataLoader(train_dataset, ...)
# valid_loader = DataLoader(valid_dataset, ...)

# ----------------------------------------------------------------------------------------------------------------------------
# STRATIFIED SPLITTING => OBIGATORY for unbalanced datasets (bad<<<good and small dataset)

#  OPTION 1
def make_weights_for_balanced_classes(images, nclasses, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    count = torch.zeros(nclasses).to(device)
    loader = DataLoader(images, batch_size=batch_size, num_workers=4)

    for _, label in tqdm(loader, desc="Counting classes"):
        label = label.to(device=device)
        idx, counts = label.unique(return_counts=True)
        count[idx] += counts

    N = count.sum()
    weight_per_class = N / count

    weight = torch.zeros(len(images)).to(device)

    for i, (img, label) in tqdm(enumerate(loader), desc="Apply weights", total=len(loader)):
        idx = torch.arange(0, img.shape[0]) + (i * batch_size)
        idx = idx.to(dtype=torch.long, device=device)
        weight[idx] = weight_per_class[label]

    return weight
    
# Batch size
batch_size = 32

# Train set
dataset_train = datasets.ImageFolder(traindir)                                                                         
                                                                                
# For unbalanced dataset we create a weighted sampler                       
weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes), batch_size)
sampler = sampler.WeightedRandomSampler(weights, len(weights))
                                                                                
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle = False, sampler = sampler, num_workers=4, pin_memory=True)     

#  OPTION 2

# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# import os

# class CustomDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.labels_df = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels_df)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
#         image = Image.open(img_name)

#         label = self.labels_df.iloc[idx, 1]

#         if self.transform:
#             image = self.transform(image)

#         return image, label

# # Path to your CSV file and image folder
# csv_file = 'path/to/labels.csv'
# image_folder = 'path/to/images'

# # Define a transform (you may need to adjust this based on your specific requirements)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Adjust size as needed
#     transforms.ToTensor(),
# ])

# # Create an instance of your custom dataset
# custom_dataset = CustomDataset(csv_file=csv_file, root_dir=image_folder, transform=transform)

# # Extract labels for stratified splitting
# labels = custom_dataset.labels_df['label'].values

# # Stratified split at the file level
# train_idx, test_idx = train_test_split(range(len(labels)), test_size=0.2, stratify=labels, random_state=42)

# # Subset dataset for training and testing
# train_dataset = torch.utils.data.Subset(custom_dataset, train_idx)
# test_dataset = torch.utils.data.Subset(custom_dataset, test_idx)

# # Create DataLoader instances
# train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)





# ----------------------------------------------------------------------------------------------------------------------------
# Prob not using this

# class ImageDataset(Dataset):
#     """Image dataset."""

#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with image info.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.image_info = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_info)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir, self.image_info.iloc[idx, 0])
#         image = io.imread(img_name)
#         id = self.image_info.iloc[idx, 1]
#         component = self.image_info.iloc[idx, 2]
#         label = "bad"

#         if self.transform:
#             image = self.transform(image)

#         return image, id, component, label