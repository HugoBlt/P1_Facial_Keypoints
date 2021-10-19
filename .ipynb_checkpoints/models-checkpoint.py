## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # convolutional layer (sees 224x224x1 image tensor)
        # (W-F)/S +1
        self.conv1 = nn.Conv2d(1, 32, 3)
        # convolutional layer (sees 111x111x32 image tensor)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # convolutional layer (sees 54x54x64 image tensor)
        self.conv3 = nn.Conv2d(64, 128, 3)
        # convolutional layer (sees 26x26x128 image tensor)
        self.conv4 = nn.Conv2d(128, 256, 3)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (256 * 12 * 12 -> 500)
        self.fc1 = nn.Linear(256 * 12 * 12, 1024)
        # linear layer (500 -> 136)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)
        # dropout layer
        self.dropout = nn.Dropout(0.25)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #import pdb
        #pdb.set_trace()
        
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4(x)))
        # flatten image input
        x = x.view(x.size(0), -1)        
        # add 1st hidden layer, with relu activation function
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.leaky_relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add 3nd hidden layer
        x = self.fc3(x)
         
        # a modified x, having gone through all the layers of your model, should be returned
        return x
