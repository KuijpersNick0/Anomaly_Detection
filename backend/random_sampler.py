import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch    
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler 

np.random.seed(0)
torch.manual_seed(0) 

class CustomSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super(CustomSubset, self).__init__(dataset, indices)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]

        if isinstance(img, torch.Tensor):
            # If the image is already a tensor, don't apply the transform
            transformed_img = img
        else:
            # Apply the transform if the image is a PIL Image or a NumPy array
            transformed_img = self.transform(img) if self.transform is not None else img

        if self.target_transform is not None:
            label = self.target_transform(label)

        return transformed_img, label


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

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(img_dataset))
    val_size = len(img_dataset) - train_size
    print(train_size)
    print(val_size)
    train_dataset, val_dataset = random_split(img_dataset, [train_size, val_size])

    target_list = torch.tensor([train_dataset.dataset.targets[i] for i in train_dataset.indices])
    print(len(target_list))
    class_count = [i for i in get_class_distribution(train_dataset.dataset).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    # Create data loaders for train and validation datasets
    # train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size, sampler=train_weighted_sampler)
    train_loader = DataLoader(dataset=CustomSubset(train_dataset.dataset, train_dataset.indices, transform=image_transforms["train"]), shuffle=False, batch_size=1, sampler=weighted_sampler)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=1)
    
    print("Length of the train_loader:", len(train_loader))
    print("Length of the val_loader:", len(val_loader))

    return train_loader, val_loader, train_size, val_size

def main(root_dir, batch_size):
    train_loader, val_loader, train_size, val_size = get_weighted_data_loaders(root_dir, batch_size)

if __name__ == "__main__": 
    root_dir = "../../data/CNN_images/Run1/" 
    batch_size = 8
    main(root_dir, batch_size)