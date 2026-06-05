import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, in_features=784, h1=16, h2=16, out_features=10):
        super().__init__() # instantiate nn.Module

        # Convolutional formula = ((Width - kernel_size + 2*padding) / stride) + 1  (floor)
        # Pooling formula = ((Width - kernel_size) / stride) + 1    (floor)

        self.conv1 = nn.Conv2d(1, 16, 3)  # out = (26 x 26 x 16)
        self.pool1 = nn.MaxPool2d(2, 2)   # out = (13 x 13 x 16)
        
        self.conv2 = nn.Conv2d(16, 32, 3) # out = (11 x 11 x 32)
        self.pool2 = nn.MaxPool2d(2, 2)   # out = (5 x 5 x 32)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(800, h1)     # in = 5*5*32 = 800
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

    