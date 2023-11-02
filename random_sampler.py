import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

from tqdm import tqdm

np.random.seed(0)
torch.manual_seed(0)  

def get_weighted_data_loaders(root_dir, batch_size):

    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
    }

    img_dataset = datasets.ImageFolder(
        root = root_dir,
        transform = image_transforms["train"]
    )  

    idx2class = {v: k for k, v in img_dataset.class_to_idx.items()}

    def get_class_distribution(dataset_obj):
        count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}

        for element in dataset_obj:
            y_lbl = element[1]
            y_lbl = idx2class[y_lbl]
            count_dict[y_lbl] += 1

        return count_dict

    target_list = torch.tensor(img_dataset.targets)
    class_count = [i for i in get_class_distribution(img_dataset).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

    class_weights_all = class_weights[target_list]

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(img_dataset))
    print(train_size)
    val_size = len(img_dataset) - train_size
    print(val_size)
    train_dataset, val_dataset = random_split(img_dataset, [train_size, val_size])

    # Create a weighted sampler for the train dataset
    train_target_list = torch.tensor(train_dataset.dataset.targets)
    train_class_weights_all = class_weights[train_target_list]
    train_weighted_sampler = WeightedRandomSampler(
        weights=train_class_weights_all,
        num_samples=len(train_class_weights_all),
        replacement=True
    )

    print(len(train_class_weights_all))
    print(len(train_dataset))

    print(len(train_target_list))
    print(len(torch.unique(train_target_list)))

    print(len(train_weighted_sampler))


    # Create a sampler for the validation dataset
    val_sampler = SubsetRandomSampler(val_dataset.indices)

    # Create data loaders for train and validation datasets
    # train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size, sampler=train_weighted_sampler)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batch_size, sampler=val_sampler)
    
    return train_loader, val_loader, train_size, val_size

def main(root_dir, batch_size):
    train_loader, val_loader, train_size, val_size = get_weighted_data_loaders(root_dir, batch_size)

if __name__ == "__main__": 
    root_dir = "../data/CNN_images/Run1/" 
    batch_size = 8
    main(root_dir, batch_size)