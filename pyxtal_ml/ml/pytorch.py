import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class torching():
    """
    A class for running PyTorch.
    """    
    def __init__(self, feature, prop, hidden_layers, test_size = 0.3):
        """
        
        """
        self.feature = feature
        self.prop = prop
        self.test_size = test_size
        self.feature_size = 1
        
        # Read the hidden layers information:
        self.n_layers, self.n_neurons = hidden_layers.values()
        
        # Perform Neural Network
        self.net = self.Net(self.feature_size, self.n_layers, self.n_neurons)
        
        optimizer = optim.SGD(self.net.parameters(), lr=0.001)
        loss_func = nn.MSELoss()  # this is for regression mean squared loss

        plt.ion()   # something about plotting

        for t in range(1000):
            prediction = self.net(self.feature)     # input x and predict based on x
            loss = loss_func(prediction, self.prop)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
            if t % 5 == 0:
                # plot and show learning process
                plt.cla()
                plt.scatter(x.data.numpy(), y.data.numpy())
                plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
                plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
                plt.pause(0.1)

        plt.ioff()
        plt.show()

    
    class Net(nn.Module):
        """
        A class for the neural network architecture defined by users.
        
        """
        def __init__(self, feature_size, n_layers, n_neurons):
            super().__init__()
            self.feature_size = feature_size
            self.n_layers = n_layers
            self.n_neurons = n_neurons
            
            if n_layers > 1:
                layers = n_layers
                if len(n_neurons) > 1:
                    self.h1 = nn.Linear(feature_size, n_neurons[0])
                    self.hidden = []
                    for i in range(1, layers):
                        self.hidden.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
                    self.predict = nn.Linear(n_neurons[layers], 1)
                else:
                    self.h1 = nn.Linear(feature_size, n_neurons[0])
                    self.hidden = []
                    for i in range(1, layers):
                        self.hidden.append(nn.Linear(n_neurons[0], n_neurons[0]))
                    self.predict = nn.Linear(n_neurons[0], 1)
                    
            else:
                self.h1 = nn.Linear(feature_size, n_neurons[0])
                self.predict = nn.Linear(n_neurons[0], 1)
                
        def forward(self, x):
            out = F.relu(self.h1(x))
#            if self.n_layers > 1:
#                for hid in self.hidden:
#                    out = hid(out)
#                    out = F.relu(out)
#            else:
#                pass
            out = self.predict(out)
            return out
            
x = autograd.Variable(torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1), requires_grad=True)  # x data (tensor), shape=(100, 1)
y = autograd.Variable(x.pow(2) + 0.2*torch.rand(x.size()), requires_grad=True)                # noisy y data (tensor), shape=(100, 1)

hl = {"n_layers": 1, "n_neurons": [10]}

testing = torching(x, y, hidden_layers = hl)