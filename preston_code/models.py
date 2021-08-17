import torch
from torch.autograd import Variable
from torch.nn import functional as F

class FFNet(torch.nn.Module):
    """Simple class to implement a feed-forward neural network in PyTorch.
    
    Attributes:
        layers: list of torch.nn.Linear layers to be applied in forward pass.
        activation: activation function to be applied between layers.
    
    """
    def __init__(self,shape,activation=None):
        """Constructor for FFNet.
        
        Arguments:
            shape: list of ints describing network shape, including input & output size.
            activation: a torch.nn function specifying the network activation.
        """
        super(FFNet, self).__init__()
        self.shape = shape
        self.layers = []
        self.activation = activation ##TODO(pculbertson): make it possible use >1 activation... maybe? who cares
        for ii in range(0,len(shape)-1):
            self.layers.append(torch.nn.Linear(shape[ii],shape[ii+1]))

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        "Performs a forward pass on x, a numpy array of size (-1,shape[0])"
        for ii in range(0,len(self.layers)-1):
            x = self.layers[ii](x)
            if self.activation:
              x = self.activation(x)

        return self.layers[-1](x)
    
class FuncNet(torch.nn.Module):
    """Simple class to implement a feed-forward neural network in PyTorch.
    
    Attributes:
        layers: list of torch.nn.Linear layers to be applied in forward pass.
        activation: activation function to be applied between layers.
    
    """
    def __init__(self,shape,activation=None):
        """Constructor for FFNet.
        
        Arguments:
            shape: list of ints describing network shape, including input & output size.
            activation: a torch.nn function specifying the network activation.
        """
        super().__init__()
        self.shape = shape
        self.activation = activation ##TODO(pculbertson): make it possible use >1 activation... maybe? who cares

        self.vars = torch.nn.ParameterList()

        for ii in range(0,len(shape)-1):
            weight_ii = torch.nn.Parameter(torch.ones(shape[ii+1], shape[ii]))
            torch.nn.init.kaiming_normal_(weight_ii)
            self.vars.append(weight_ii)

            bias_ii = torch.nn.Parameter(torch.zeros(shape[ii+1]))
            self.vars.append(bias_ii)

    def forward(self, x, vars=None):
        "Performs a forward pass on x, a numpy array of size (-1,shape[0])"
        if vars is None:
            vars = self.vars

        idx = 0
        for ii in range(len(self.shape)-2):
            w_ii, b_ii = vars[idx], vars[idx+1]
            x = F.linear(x, w_ii, b_ii)

            if self.activation:
                x = F.relu(x, inplace=True)
                idx += 2

        w_ii, b_ii = vars[idx], vars[idx+1]
        x = F.linear(x, w_ii, b_ii)
        return x


class SkipNet(torch.nn.Module):
    """
    A feedforward neural network with skip connections.
    """
    def __init__(self,shape,skips,activation=None):
        """Constructor for FFNet.
        
        Arguments:
            shape: list of ints describing network shape, including input & output size.
            activation: a torch.nn function specifying the network activation.
        """
        super(SkipNet, self).__init__()
        self.shape = shape
        self.skips = skips
        self.nx = shape[0]
        if any([(ii < 1) or (ii > len(shape)-2) for ii in skips]):
            raise ValueError
        
        self.layers = []
        self.activation = activation ##TODO(pculbertson): make it possible use >1 activation... maybe? who cares
        for ii in range(0,len(shape)-1):
            if ii in self.skips:
                self.shape[ii] += self.nx
            self.layers.append(torch.nn.Linear(shape[ii],shape[ii+1]))

        self.layers = torch.nn.ModuleList(self.layers)
        
    def forward(self, x):
        "Performs a forward pass on x, a numpy array of size (-1,shape[0])"
        y = x
        for ii in range(0,len(self.layers)-1):
            if ii in self.skips:
                y = torch.cat((y,x),axis=-1)
            y = self.layers[ii](y)
            if self.activation:
              y = self.activation(y)

        return self.layers[-1](y)
    
class SIREN(torch.nn.Module):
    """Simple class to implement a SIREN neural network in PyTorch.
    
    Attributes:
        layers: list of torch.nn.Linear layers to be applied in forward pass.
    
    """
    def __init__(self,shape, w_0=30.):
        """Constructor for SIREN.
        
        Arguments:
            shape: list of ints describing network shape, including input & output size.
            activation: a torch.nn function specifying the network activation.
        """
        super(SIREN, self).__init__()
        self.shape = shape
        self.layers = []
        self.activation = torch.sin
        self.w_0 = w_0
        
        for ii in range(0,len(shape)-1):
            self.layers.append(torch.nn.Linear(shape[ii],shape[ii+1]))
            c = torch.sqrt(torch.tensor(6/shape[ii]))
            inv_sqrt_n = torch.sqrt(torch.tensor(1/shape[ii]))
            torch.nn.init.uniform_(
                self.layers[ii].weight, -c, c)
            torch.nn.init.uniform_(
                self.layers[ii].bias, -inv_sqrt_n, inv_sqrt_n)

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        "Performs a forward pass on x, a numpy array of size (-1,shape[0])"
        for ii in range(0,len(self.layers)-1):
            x = self.w_0*x@self.layers[ii].weight.T + self.layers[ii].bias
            if self.activation:
              x = self.activation(x)

        return self.layers[-1](x)
    
class RBF(torch.nn.Module):
    def __init__(self, num_kernels, in_dim, out_dim):
        super(RBF, self).__init__()
        
        self.n = in_dim
        self.num_kernels = num_kernels
        self.out_dim = out_dim
        
        self.centers = torch.nn.Parameter(
            torch.randn(1,self.num_kernels,self.n), requires_grad=True)
        
        self.length_scales = torch.nn.Parameter(
            0.1*torch.randn(1,self.num_kernels,1), requires_grad=True)
        
        self.weights = torch.nn.Parameter(
            torch.randn(1,self.out_dim,self.num_kernels), requires_grad=True)
        
    def forward(self, x):
        kernel_vals = torch.exp(
            -(1/torch.square(self.length_scales))
            * torch.mean(torch.square(self.centers - x.reshape(-1,1,self.n)),
                         dim=-1, keepdim=True))
        return torch.matmul(self.weights,kernel_vals).squeeze(-1)