import torch
import torch.nn as nn


class MyFirstHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(MyFirstHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)
