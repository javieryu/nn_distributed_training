import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):
    """Implements a basic convolutional neural network with one
    convolutional layer and two subsequent linear layers for the MNIST
    classification problem.
    """

    def __init__(self, num_filters, kernel_size, linear_width):
        super(MNISTConvNet, self).__init__()
        self.conv = nn.Conv2d(1, num_filters, kernel_size)
        conv_out_size = (28 - (kernel_size - 1)) ** 2
        self.fc1 = nn.Linear(conv_out_size, linear_width)
        self.fc2 = nn.Linear(linear_width, 10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x