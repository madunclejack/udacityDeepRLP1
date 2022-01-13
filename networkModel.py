# Defines the Deep Neural Network used for learning

import torch
import torch.nn as nn
import torch.nn.functional as F

class agentQNetwork(nn.Module):
    def __init__(self, numStates, numActions, hiddenLayers=[64, 64], seed=343):
        
        #numStates: (int) number of input parameters
        #numActions: (int)number of possible outputs
        #hiddenLayers: (int array)  specifies the number and size of hidden layers
        #seed: (int)random seed
        
        super(agentQNetwork, self).__init__()
        
        # Set up inputs and outputs
        self.inputSize = numStates
        self.outputSize = numActions
        self.hiddenLayers = hiddenLayers
        
        self.seed = torch.manual_seed(seed)

        
        # Set up the network
        self.neuralNet = None
        self.output = None
        self.buildNetwork()
    
    def buildNetwork(self):
        # Create first Input layer using a Module List
        self.neuralNet = nn.ModuleList([nn.Linear(self.inputSize, self.hiddenLayers[0])])
        
        # Make a set of pairs describing input and output sizes for each hidden layer
        layerSizes = zip(self.hiddenLayers[:-1], self.hiddenLayers[1:])
        
        # Extend the neural net
        self.neuralNet.extend([nn.Linear(hiddenIn, hiddenOut) for hiddenIn, hiddenOut in layerSizes])
        
        # Define the output layer
        self.output = nn.Linear(self.hiddenLayers[-1], self.outputSize)
    
    def forward(self, state):
        
        #Determine the steps for a forward pass through the network.
        #Return: final state
        
        for layer in self.neuralNet:
            state = F.relu(layer(state))
        state = self.output(state)
        
        return state