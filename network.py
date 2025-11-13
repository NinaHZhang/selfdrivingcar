"""
this file contains a neural network module for us to define our actor and critic networks in PPO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        '''Initialize the network and set up the layers.
        Parameters:
            in_dim - input dimensions
            out_dim - output dimensions as an int
            
        Return:
            None'''
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 32) #32 neurons a layer instead of 64
        self.layer2 = nn.Linear(32, out_dim)
    
    def forward(self, obs):
        '''Runs a forward pass on the NN.
        Parameters:
            obs - observation to pass as input
        
        Return:
            output - the output of forward pass
        '''
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        activation1 = F.relu(self.layer1(obs)) # first layer with relu activation
        output = self.layer2(activation1) #output layer with no activation

        return output

''' 
  to test the setup, use this. it should output a tensor with 2 values
if __name__ == "__main__":
    network = FeedForwardNN(in_dim=6, out_dim=2)
    test_obs = np.array([1.0, 0.5, -0.3, 0.8, 0.2, 0.1])
    output = network.forward(test_obs)
    print("Output:", output)
    ''' 