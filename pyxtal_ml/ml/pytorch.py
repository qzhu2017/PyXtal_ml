import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import torch.nn.functional as F


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
        
        # Building Neural Network architecture using Net class
        self.model = self.Net(self.feature_size, self.n_layers, self.n_neurons)
        
        # Learning parameter for NN
        optimizer = optim.SGD(self.model.parameters(), lr = 0.005)
        loss_func = nn.MSELoss()  # mean squared eror loss

        # Learning step
        for t in range(1000):
            optimizer.zero_grad()
            
            y_ = self.model(self.feature)
            loss = loss_func(y_, y)
            
            loss.backward(retain_graph=True)
            optimizer.step()
            
#            prediction = self.net(self.feature)     # input x and predict based on x
#            loss = loss_func(prediction, self.prop)     # must be (1. nn output, 2. target)
#
#            optimizer.zero_grad()   # clear gradients for next train
#            loss.backward(retain_graph=True)         # backpropagation, compute gradients
#            optimizer.step()        # apply gradients
            
            if t % 5 == 0:
                # plot and show learning process
                plt.cla()
                plt.scatter(x.data.numpy(), y.data.numpy())
                plt.plot(x.data.numpy(), y_.data.numpy(), 'r-', lw=5)
                plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
                plt.pause(0.1)

        plt.ioff()
        plt.show()
        
        # Eval
        
#        self.model.eval()
#        with torch.no_grad():
#            y_ = self.model(self.feature)
        

    
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
                if len(n_neurons) > 1:              # different sizes of neurons in layers
                    self.h1 = nn.Linear(feature_size, n_neurons[0])
                    self.hidden = []
                    for i in range(n_layers-1):
                        self.hidden.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
                    self.predict = nn.Linear(n_neurons[n_layers-1], 1)
                    
                else:                               # same size of neurons in layers
                    self.h1 = nn.Linear(feature_size, n_neurons[0])
                    self.hidden = []
                    for i in range(n_layers-1):
                        self.hidden.append(nn.Linear(n_neurons[0], n_neurons[0]))
                    self.predict = nn.Linear(n_neurons[0], 1)
                    
            else:
                self.h1 = nn.Linear(feature_size, n_neurons[0])
                self.predict = nn.Linear(n_neurons[0], 1)
                
        def forward(self, x):
            out = F.relu(self.h1(x))
            if self.n_layers > 1:
                for hid in self.hidden:
                    out = hid(out)
                    out = F.relu(out)
            else:
                pass
            out = self.predict(out)
            return out
            
x = torch.unsqueeze(torch.linspace(-1, 1, 100, requires_grad=True), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                # noisy y data (tensor), shape=(100, 1)

hl = {"n_layers": 1, "n_neurons": [10]}

testing = torching(x, y, hidden_layers = hl)