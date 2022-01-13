# -*- coding: utf-8 -*-

#Import libraries

import numpy as np
import random
import os 
import torch
import torch.nn
import torch.nn.functional as fn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#architecture of neural network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size=input_size
        self.nb_action=nb_action
        self.fc1=nn.Linear(input_size, 30)#size of first layer, size of second layer(hidden)
        self.fc2=nn.Linear(30, nb_action)
        
    def forward(self, state):
        x = fn.relu(self.fc1(state))
        q_values=self.fc2(x)
        return q_values
    
#experience replay 
        
