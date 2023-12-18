import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from imblearn.over_sampling import SMOTE

# Define the paths to the image folders
image_folder_path = '/home/nick-kuijpers/Documents/Railnova/data/CNN_images/Run2/CNN_J2' 

# Create the ImageFolder datasets
dataset = ImageFolder(root=image_folder_path , transform=ToTensor())  

# Assuming X_train is your flattened image data and y_train is the corresponding labels
X_train = torch.stack([sample[0] for sample in dataset])
y_train = torch.tensor([sample[1] for sample in dataset])

# Assuming original_shape is the shape of your original images
original_shape = X_train.shape[1:]
X_train = X_train.reshape(-1, original_shape[0], original_shape[1], original_shape[2])

# Check class distribution before applying SMOTE
print("Class distribution before SMOTE:", torch.bincount(y_train))

# Assuming X_train and y_train are your flattened image data and labels
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train.reshape(-1, original_shape[0] * original_shape[1] * original_shape[2]), y_train)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Assuming original_shape is the shape of your original images
X_resampled = X_resampled.reshape(-1, original_shape[0], original_shape[1], original_shape[2])

# Check class distribution after applying SMOTE
print("Class distribution after SMOTE:", torch.bincount(y_resampled))
