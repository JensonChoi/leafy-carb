import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, dropout_pct=0):
        super().__init__()
        tempmodel = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(tempmodel.children())[:-1]))
          
        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_pct)
        self.bnfc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        x = torch.squeeze(x)  # flatten
        x = self.relu(self.fc1(x))
        x = self.bnfc1(self.dropout(x))
        x = self.fc2(x)
        return x
