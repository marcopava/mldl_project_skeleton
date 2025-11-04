import torch
from torch import nn
from .backbones import MyFirstBackbone
from .heads import MyFirstHead


# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(
            160000, 200
        )  # 200 is the number of classes in TinyImageNet
        self.conv_stack = nn.Sequential(
            self.conv1, self.conv2, self.conv3, nn.ReLU(), nn.Flatten(), self.fc1
        )

    def forward(self, x):
        # Define forward pass

        logits = self.conv_stack(x)

        return logits
