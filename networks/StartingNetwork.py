import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        tempmodel = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.model = torch.nn.Sequential(*(list(tempmodel.children())[:-1]))
        try: # Check if GPU is available
            device = torch.device('cuda')
            (self.model).to(device)
        except:
            print("Couldnt move model to GPU :(")
          
        self.fc1 = nn.Linear(512, 512)
        self.bnfc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        x = torch.squeeze(x)
        x = self.bnfc1(nn.ReLU(self.fc1(features)))
        x = self.fc2(x)
        return x
