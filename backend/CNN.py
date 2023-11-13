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

from random_sampler import get_weighted_data_loaders

# from numba import cuda 

# device.reset()
# device = cuda.get_current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Import loaders 
train_loader, valid_loader, train_size, valid_size = get_weighted_data_loaders("../../data/CNN_images/Run1/", 8)

# applying transforms to the data
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# number of classes
num_classes = 17

# load pretrained resnet50, 152 bigger but better perf should try
# Maybe try DenseNet after this :DenseNet161_Weights.IMAGENET1K_V1
resnet_50 = models.resnet50(weights="IMAGENET1K_V2")

# Freeze model parameters, coz we are fine-tuning
for param in resnet_50.parameters():
  param.requires_grad = False

# change the final layer of Resnet50 Model for fine-tuning
fc_inputs = resnet_50.fc.in_features

resnet_50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4), 
    nn.Linear(256, 17), # since we have 17 classes
    nn.LogSoftmax(dim=1) # for using NLLLoss()
)

# convert model to GPU
resnet_50 = resnet_50.to(device)

# define optimizer and loss function
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet_50.parameters())

def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels.long())
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_size 
        avg_train_acc = train_acc/train_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_size 
        avg_valid_acc = valid_acc/valid_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        # Save if the model has best accuracy till now
        # torch.save(model, 'model_'+str(epoch)+'.pt')
            
    return model, history

num_epochs = 25
trained_model, history = train_and_validate(resnet_50, loss_func, optimizer, num_epochs)
torch.save(history, 'history.pt')

torch.save(trained_model,'trained_model.pt')

history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.savefig('loss_curve.png')
plt.show()

plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.savefig('_accuracy_curve.png')
plt.show()

# def computeTestSetAccuracy(model, loss_criterion):
#     '''
#     Function to compute the accuracy on the test set
#     Parameters
#         :param model: Model to test
#         :param loss_criterion: Loss Criterion to minimize
#     '''

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     test_acc = 0.0
#     test_loss = 0.0

#     # Validation - No gradient tracking needed
#     with torch.no_grad():

#         # Set to evaluation mode
#         model.eval()

#         # Validation loop
#         for j, (inputs, labels) in enumerate(test_data):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             # Forward pass - compute outputs on input data using the model
#             outputs = model(inputs)

#             # Compute loss
#             loss = loss_criterion(outputs, labels)

#             # Compute the total loss for the batch and add it to valid_loss
#             test_loss += loss.item() * inputs.size(0)

#             # Calculate validation accuracy
#             ret, predictions = torch.max(outputs.data, 1)
#             correct_counts = predictions.eq(labels.data.view_as(predictions))

#             # Convert correct_counts to float and then compute the mean
#             acc = torch.mean(correct_counts.type(torch.FloatTensor))

#             # Compute total accuracy in the whole batch and add to valid_acc
#             test_acc += acc.item() * inputs.size(0)

#             print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

#     # Find average test loss and test accuracy
#     avg_test_loss = test_loss/test_data_size 
#     avg_test_acc = test_acc/test_data_size

#     print("Test accuracy : " + str(avg_test_acc))

# computeTestSetAccuracy(trained_model, loss_func)

def train_CNN(images):
    return None

def main(images):
    return None