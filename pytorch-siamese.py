# I will build a siamese network to detect if two face pictures are from the same person or not, from the Labeled Faces in the Wild (LFW) dataset.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models
from PIL import Image

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

print("Is cuda available ? ", torch.cuda.is_available())
torch.cuda.empty_cache()

# Hyperparameters

BATCH_SIZE = 32
NUMB_EPOCHS = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
    
# Image size
IMG_SIZE = 64

# Custom Dataset classes
        
class trainingDataset(Dataset):
        def __init__(self, imageFolderDataset, data, transform=None):
            self.imageFolderDataset=imageFolderDataset
            self.data=data
            self.transform=transform
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            image1 = os.path.join(self.imageFolderDataset, self.data.iloc[idx, 0] + '.jpg')
            image2 = os.path.join(self.imageFolderDataset, self.data.iloc[idx, 1] + '.jpg')
            
            image1 = Image.open(image1)
            image2 = Image.open(image2)
            
            is_paired = self.data.iloc[idx, 2]
            is_paired = torch.tensor(is_paired).float()
            
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
            return image1, image2, is_paired

class testDataset(Dataset):
        def __init__(self, imageFolderDataset, data, transform=None):
            self.testFolderDataset=imageFolderDataset
            self.data=data
            self.transform=transform
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            image1 = os.path.join(self.testFolderDataset, self.data.iloc[idx, 0] + '.jpg')
            image2 = os.path.join(self.testFolderDataset, self.data.iloc[idx, 1] + '.jpg')
            
            image1 = Image.open(image1)
            image2 = Image.open(image2)
            
            is_paired = self.data.iloc[idx, 2]
            is_paired = torch.tensor(is_paired)
            
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
                
            return image1, image2, is_paired

def get_data(BATCH_SIZE, IMG_SIZE):
    # Load data
    all_data = pd.read_csv('data.csv')
    
    # For testing purposes, we will only use a small fraction of the data
    #all_data = all_data.sample(frac=0.1, random_state=0)

    # Split data into train and test sets
    train_data = all_data.sample(frac=0.7, random_state=0)
    val_data = all_data.drop(train_data.index)
    test_data = val_data.sample(frac=0.3, random_state=0)
    val_data = val_data.drop(test_data.index)

    # Build a custom dataset            

    train_set = trainingDataset(imageFolderDataset=os.path.join('archive', 'lfw-deepfunneled', 'lfw-deepfunneled'),
                                data = train_data,
                                transform=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                            transforms.ToTensor() ]))

    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=6)

    val_set = trainingDataset(imageFolderDataset=os.path.join('archive', 'lfw-deepfunneled', 'lfw-deepfunneled'),
                            data = val_data,
                            transform=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                            transforms.ToTensor() ]))

    valloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=6)

    vis_dataloader = DataLoader(train_set,
                            shuffle=True,
                            batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    plt.figure(1)
    plt.imshow(torchvision.utils.make_grid(concatenated).permute(1, 2, 0))
    plt.title('Labels: ' + str(example_batch[2].numpy()))
    plt.savefig(os.path.join('media', 'example_batch.png'))

    test_set = testDataset(imageFolderDataset=os.path.join('archive', 'lfw-deepfunneled', 'lfw-deepfunneled'),
                        data = test_data,
                        transform=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                        transforms.ToTensor() ]))

    testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=6)
    
    return trainloader, valloader, testloader

# Model

class SiameseNetwork(nn.Module):
        def __init__(self):
            super(SiameseNetwork, self).__init__()
            
            self.cnn1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(3, 64, kernel_size=3, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Dropout2d(p=0.2),
                
                nn.ReflectionPad2d(1),
                nn.Conv2d(64, 64, kernel_size=3, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Dropout2d(p=0.2),
                
                nn.ReflectionPad2d(1),
                nn.Conv2d(64, 32, kernel_size=3, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.Dropout2d(p=0.2),
            )
            
            self.fc1 = nn.Sequential(
                nn.Linear(2*32*IMG_SIZE*IMG_SIZE, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
        def forward_once(self, x):
            output = self.cnn1(x)
            output = output.view(output.size()[0], -1)
            
            return output
            
        def forward(self, input1, input2):
            #Image 1
            output1 = self.forward_once(input1)
            
            # Image 2
            output2 = self.forward_once(input2)
            
            # Concatenate outputs
            output = torch.cat((output1, output2), 1)
            output = self.fc1(output)
            
            return output
        
# Build the model

def get_model(LEARNING_RATE, MOMENTUM):
    
    network = SiameseNetwork().cuda()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    return network, criterion, optimizer, scheduler

# Run everything

def main(BATCH_SIZE, NUMB_EPOCHS, LEARNING_RATE, MOMENTUM, IMG_SIZE, fig=2):
    
    # Load data
    trainloader, valloader, testloader = get_data(BATCH_SIZE, IMG_SIZE)

    # Build the model
    net, criterion, optimizer, scheduler = get_model(LEARNING_RATE, MOMENTUM)

    # Train the network
    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(1, NUMB_EPOCHS+1):
        for i, data in enumerate(trainloader, 0):
            img0, img1, labels = data # Image shape: (batch_size, channel, IMG_SIZE, IMG_SIZE), labels shape: (batch_size)
            img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda() # Move to GPU
            optimizer.zero_grad() # Clear gradients
            outputs = net(img0, img1).squeeze() # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
        
        scheduler.step()
                
        # Plot loss history
        correct = 0
        total = 0

        with torch.no_grad():
            for data in valloader:
                img0, img1, labels = data
                img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()
                outputs = net(img0, img1).squeeze()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
        print("Epoch: ", epoch)     
        print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))
                
        if epoch % 8 == 0 or epoch == NUMB_EPOCHS:
            torch.save({
                'epoch': NUMB_EPOCHS,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join('models', 'model_' + str(epoch) + '_EP.pt'))
            plt.figure(fig)
            plt.plot(counter, loss_history)
            plt.title('Loss History at epoch: ' + str(epoch))
            plt.savefig(os.path.join('media', 'loss_history_' + str(epoch) + '.png'))
            fig += 1
            
    print('Finished Training')

    # Test the network
    with torch.no_grad():
        for data in testloader:
            img0, img1, labels = data
            img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()
            outputs = net(img0, img1).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    main(BATCH_SIZE, NUMB_EPOCHS, LEARNING_RATE, MOMENTUM, IMG_SIZE)