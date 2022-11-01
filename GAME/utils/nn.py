# Author: Minh Hua
# Date: 10/31/2022
# Purpose: This module contains Neural Network classes for learning the transition models.

import torch.nn as nn

class LinearNeuralNet(nn.Module):
    """Fully connected neural network with one hidden layer"""
    def __init__(self, input_size, hidden_size, output_size, activation_fnc:str='relu') -> None:
        """
        Description:
            Initializes a one-layer neural network with linear output.

        Arguments:
            input_size: the number of input nodes.
            hidden_size: the number of hidden nodes.
            output_size: the number of output nodes.
            activation_fnc: the activation function. Must be ['sigmoid', 'relu'].

        Return:
            (None)
        """
        super(LinearNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        if activation_fnc == 'relu':
            self.activation_fnc = nn.ReLU()
        elif activation_fnc == 'sigmoid':
            self.activation_fnc = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x) -> list:
        """
        Description:
            Feedforward the input x.

        Arguments:
            x: the input to feed through the network.

        Return:
            (list) the output vector
        """
        out = self.l1(x)
        out = self.activation_fnc(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out