import torch.nn as nn


# Define the backbone module
class MyFirstBackbone(nn.Module):
    def __init__(self):
        super(MyFirstBackbone, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        # output 224x224x64

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=2, stride=3)
        # output 76x76x128

        self.conv3 = nn.Conv2d(128, 256, kernel_size=6, padding=1, stride=3)
        # output 24x24x256

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.ReLU(self.conv3(x))

        return x
