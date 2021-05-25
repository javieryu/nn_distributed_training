""" Tools for sine activated neural networks.
SIREN Paper: https://arxiv.org/pdf/2006.09661.pdf

Related codebase: https://github.com/GlassyWing/fourier-feature-networks

Written by: Javier Yu (May 25, 2021)
"""
import torch
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)


class SIRENLayer(torch.nn.Module):
    """Sine activated linear layer with a scaling factor.
    Forward pass is f(x) = sin(scale * W @ x).
    """

    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features)
        self.scale = scale
        self.init_weights()

    def init_weights(self):
        """Initialize layer weights following SIREN paper suggestions."""
        c = np.sqrt(6 / self.out_features)
        with torch.no_grad():
            self.linear.weight.uniform_(-c, c)

    def forward(self, x):
        x = self.linear(x)
        return torch.sin(self.scale * x)


class FourierNet(torch.nn.Module):
    """A feed-forward neural network with a fourier feature (siren) first
    layer, ReLU activated middle layers, and sigmoid activated final layer.
    """

    def __init__(self, shape, scale=1.0):
        super().__init__()

        layers = []

        for i in range(len(shape) - 1):
            if i == 0:
                layers.append(SIRENLayer(shape[0], shape[1], scale=scale))
            else:
                layers.append(torch.nn.Linear(shape[i], shape[i + 1]))

            if i != len(shape) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
            else:
                layers.append(torch.nn.Sigmoid())

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
