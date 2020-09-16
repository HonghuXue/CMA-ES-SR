# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:16:41 2020

@author: hongh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 512)
        self.l3 = nn.Linear(512, 1024)
        self.l4 = nn.Linear(1024, 1)
        self.prelu = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(num_features=input_dim)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        
    def forward(self, x):
#            q = self.prelu(self.l1(x))
#            q = self.prelu(self.l2(q))
        q = self.prelu(self.l1(self.bn1(x)))
        q = self.prelu(self.l2(self.bn2(q)))
        q = self.prelu(self.l3(self.bn3(q)))
        q = self.l4(q)
        return q  

