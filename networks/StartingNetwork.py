import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28 * 48, 256)
        self.bnfc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.pool1(x)
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.pool2(x)
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.bnfc1(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
