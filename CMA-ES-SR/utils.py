# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:14:57 2020

@author: hongh
"""
import torch
import numpy as np
import torch.nn as nn

class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.size = 0
    
    def add(self, transition):
        self.size +=1
        # transiton is tuple of (parameter, cost)
        self.buffer.append(transition)
    
    def sample(self, batch_size):            
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, cost = [], []            
        for i in indexes:
            s, c = self.buffer[i]
            state.append(np.array(s, copy=False))
            cost.append(np.array(c, copy=False))            
        return np.array(state), np.array(cost)

    def sequential_sample(self, batch_size):            
        state, cost = [], []            
        for i in range(batch_size):
            s, c = self.buffer[i]
            state.append(np.array(s, copy=False))
            cost.append(np.array(c, copy=False))            
        return np.array(state), np.array(cost)
    
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)