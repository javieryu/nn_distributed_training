import torch.nn as nn


class MNISTConvNet(nn.Module):
    """Implements a basic convolutional neural network with one
    convolutional layer and two subsequent linear layers for the MNIST
    classification problem.
    """

    def __init__(self, num_filters, kernel_size, linear_width):
        super().__init__()
        conv_out_width = 28 - (kernel_size - 1)
        pool_out_width = int(conv_out_width / 2)
        fc1_indim = num_filters * (pool_out_width ** 2)

        self.seq = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(fc1_indim, linear_width),
            nn.ReLU(inplace=True),
            nn.Linear(linear_width, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.seq(x)
