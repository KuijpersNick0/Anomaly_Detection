# import libraries
import torch
from torchvision import models, datasets
from torchvision import transforms
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import json

image_transforms = {
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

img_dataset = datasets.ImageFolder(
    root = "../../data/CNN_images/Run1/",
    transform = image_transforms["test"]
)  

idx2class = {v: k for k, v in img_dataset.class_to_idx.items()}

# number of classes
num_classes = 17


def predict(model, image_path):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    
    transform = image_transforms['test']

    test_image = Image.open(image_path) 
    
    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)
        predictions = []
        for i in range(3):
            # print("Prediction", i+1, ":", idx2class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
            prediction = {
                "class": idx2class[topclass.cpu().numpy()[0][i]],
                "score": float(topk.cpu().numpy()[0][i])
            }
            predictions.append(prediction)
        return json.dumps(predictions)
        

# model = torch.load('/home/nick-kuijpers/Documents/Railnova/Python/backend/models/trained_model.pt') #GPU only
model = torch.load('/home/nick-kuijpers/Documents/Railnova/Python/backend/models/trained_model.pt',map_location ='cpu')
# predict(model, '../../data/CNN_images/Run1/J2/J2_2F3_G12.jpg') #Confondu avec U600
# predict(model, '../../data/CNN_images/Run1/U600/U600_2F2_G244.jpg') # ok avec U600
# predict(model, '../../data/CNN_images/Run1/U911U5/U911U5_2F2_G219.jpg') # Confonds encore avec du U600...
# predict(model, '../../data/CNN_images/Run1/U701/U701_2F2_G156.jpg') # U904U3...

def main():
    # Load the model
    model = torch.load('/home/nick-kuijpers/Documents/Railnova/Python/backend/models/trained_model.pt', map_location='cpu')

    # Call the predict function with a test image
    predict(model, '../../data/CNN_images/Run1/J2/J2_2F3_G12.jpg')

if __name__ == '__main__':
    main()
