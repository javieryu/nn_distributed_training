from torch import nn
import numpy as np
import torch


class FFReLUNet(nn.Module):
    """
    Implements a basic feed forward neural network that uses
    ReLU activations for all of the layers.
    """

    def __init__(self, shape):
        """Constructor for network.

        Args:
            shape (list of ints): list of network layer shapes, which
            includes the input and output layers.
        """
        super(FFReLUNet, self).__init__()
        self.shape = shape

        # Build up the layers
        layers = []
        # (1, 64, 64, 1)
        for i in range(len(shape) - 1):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            if i != (len(shape) - 2):
                layers.append(nn.ReLU(inplace=True))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass on the input through the network.

        Args:
            x (torch.Tensor): Input tensor dims [batch, self.shape[0]]

        Returns:
            torch.Tensor: Output of network. [batch, self.shape[-1]]
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        return self.seq(x)
