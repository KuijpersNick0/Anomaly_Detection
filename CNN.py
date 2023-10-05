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

# # applying transforms to the data
# image_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(size=256, scale=(0.8,1.0)),
#         transforms.RandomRotation(degrees=15),
#         transforms.RandomHorizontalFlip(),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ]),
#     'valid': transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])
# }

# # Load data
# # Set train, valid, and test directory
# train_directory = 'food-11k-sub/train'
# valid_directory = 'food-11k-sub/valid'
# test_directory = 'food-11k-sub/test'

# # batch size
# bs = 32

# # number of epochs
# epochs = 20

# # number of classes
# num_classes = 11

# # device 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Load data from directory
# data = {
#     'train': datasets.ImageFolder(root=train_directory,
#                                   transform=image_transforms['train']),
#     'valid': datasets.ImageFolder(root=valid_directory,
#                                   transform=image_transforms['valid']),
#     'test': datasets.ImageFolder(root=test_directory,
#                                  transform=image_transforms['test'])
# }

# # Get a mapping of the indices to the class names, in order to see the output classes of the test images.
# idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
# print(idx_to_class)

# # size of data, to be used for calculating Averge Loss and Accuracy
# train_data_size = len(data['train'])
# valid_data_size = len(data['valid'])
# test_data_size = len(data['test'])

# # Create iterators for the Data loaded using DataLoader module
# train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)
# valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=True)
# test_data = DataLoader(data['test'], batch_size=bs, shuffle=True)

# train_data_size, valid_data_size, test_data_size

# # load pretrained resnet50, 152 bigger but better perf should try
# # Maybe try DenseNet after this :DenseNet161_Weights.IMAGENET1K_V1
# resnet_50 = models.resnet50(weights="IMAGENET1K_V2")

# # Freeze model parameters, coz we are fine-tuning
# for param in resnet_50.parameters():
#   param.requires_grad = False

# # change the final layer of Resnet50 Model for fine-tuning
# fc_inputs = resnet_50.fc.in_features

# resnet_50.fc = nn.Sequential(
#     nn.Linear(fc_inputs, 256),
#     nn.ReLU(),
#     nn.Dropout(0.4), 
#     nn.Linear(256, 2), # since we have 2 classes: Bad, Good
#     nn.LogSoftmax(dim=1) # for using NLLLoss()
# )

# # convert model to GPU
# resnet_50 = resnet_50.to(device)

# # define optimizer and loss function
# loss_func = nn.NLLLoss()
# optimizer = optim.Adam(resnet_50.parameters())

# def train_and_validate(model, loss_criterion, optimizer, epochs=25):
#     '''
#     Function to train and validate
#     Parameters
#         :param model: Model to train and validate
#         :param loss_criterion: Loss Criterion to minimize
#         :param optimizer: Optimizer for computing gradients
#         :param epochs: Number of epochs (default=25)
  
#     Returns
#         model: Trained Model with best validation accuracy
#         history: (dict object): Having training loss, accuracy and validation loss, accuracy
#     '''
    
#     start = time.time()
#     history = []
#     best_acc = 0.0

#     for epoch in range(epochs):
#         epoch_start = time.time()
#         print("Epoch: {}/{}".format(epoch+1, epochs))
        
#         # Set to training mode
#         model.train()
        
#         # Loss and Accuracy within the epoch
#         train_loss = 0.0
#         train_acc = 0.0
        
#         valid_loss = 0.0
#         valid_acc = 0.0
        
#         for i, (inputs, labels) in enumerate(train_data):

#             inputs = inputs.to(device)
#             labels = labels.to(device)
            
#             # Clean existing gradients
#             optimizer.zero_grad()
            
#             # Forward pass - compute outputs on input data using the model
#             outputs = model(inputs)
            
#             # Compute loss
#             loss = loss_criterion(outputs, labels)
            
#             # Backpropagate the gradients
#             loss.backward()
            
#             # Update the parameters
#             optimizer.step()
            
#             # Compute the total loss for the batch and add it to train_loss
#             train_loss += loss.item() * inputs.size(0)
            
#             # Compute the accuracy
#             ret, predictions = torch.max(outputs.data, 1)
#             correct_counts = predictions.eq(labels.data.view_as(predictions))
            
#             # Convert correct_counts to float and then compute the mean
#             acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
#             # Compute total accuracy in the whole batch and add to train_acc
#             train_acc += acc.item() * inputs.size(0)
            
#             #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
#         # Validation - No gradient tracking needed
#         with torch.no_grad():

#             # Set to evaluation mode
#             model.eval()

#             # Validation loop
#             for j, (inputs, labels) in enumerate(valid_data):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # Forward pass - compute outputs on input data using the model
#                 outputs = model(inputs)

#                 # Compute loss
#                 loss = loss_criterion(outputs, labels)

#                 # Compute the total loss for the batch and add it to valid_loss
#                 valid_loss += loss.item() * inputs.size(0)

#                 # Calculate validation accuracy
#                 ret, predictions = torch.max(outputs.data, 1)
#                 correct_counts = predictions.eq(labels.data.view_as(predictions))

#                 # Convert correct_counts to float and then compute the mean
#                 acc = torch.mean(correct_counts.type(torch.FloatTensor))

#                 # Compute total accuracy in the whole batch and add to valid_acc
#                 valid_acc += acc.item() * inputs.size(0)

#                 #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
#         # Find average training loss and training accuracy
#         avg_train_loss = train_loss/train_data_size 
#         avg_train_acc = train_acc/train_data_size

#         # Find average training loss and training accuracy
#         avg_valid_loss = valid_loss/valid_data_size 
#         avg_valid_acc = valid_acc/valid_data_size

#         history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
#         epoch_end = time.time()
    
#         print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
#         # Save if the model has best accuracy till now
#         # torch.save(model, 'model_'+str(epoch)+'.pt')
            
#     return model, history

# num_epochs = 25
# trained_model, history = train_and_validate(resnet_50, loss_func, optimizer, num_epochs)
# torch.save(history, 'history.pt')

# torch.save(trained_model,'trained_model.pt')

# history = np.array(history)
# plt.plot(history[:,0:2])
# plt.legend(['Tr Loss', 'Val Loss'])
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.ylim(0,1)
# plt.savefig('loss_curve.png')
# plt.show()

# plt.plot(history[:,2:4])
# plt.legend(['Tr Accuracy', 'Val Accuracy'])
# plt.xlabel('Epoch Number')
# plt.ylabel('Accuracy')
# plt.ylim(0,1)
# plt.savefig('_accuracy_curve.png')
# plt.show()

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

# def predict(model, test_image_name):
#     '''
#     Function to predict the class of a single test image
#     Parameters
#         :param model: Model to test
#         :param test_image_name: Test image

#     '''
    
#     transform = image_transforms['test']

#     test_image = Image.open(test_image_name)
#     plt.imshow(test_image)
    
#     test_image_tensor = transform(test_image)

#     if torch.cuda.is_available():
#         test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
#     else:
#         test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
#     with torch.no_grad():
#         model.eval()
#         # Model outputs log probabilities
#         out = model(test_image_tensor)
#         ps = torch.exp(out)
#         topk, topclass = ps.topk(3, dim=1)
#         for i in range(3):
#             print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])

# model = torch.load('trained_model.pt')
# predict(model, 'food-11k-sub/test/Egg/3_22.jpg')



# def train_CNN(images):
#     return None

# def main(images):
#     return None