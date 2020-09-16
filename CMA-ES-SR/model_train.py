# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:20:18 2020

@author: hongh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math as ms
import numpy as np
from utils import init_weights, ReplayBuffer
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class Model_learn:                          
    def __init__(self, Model, device):
        self.loss = nn.MSELoss(reduction='mean')      
        self.device = device
        self.model = Model.to(device)
        self.max_training_iteration = 100
#        self.model.apply(init_weights)
        
    def model_prediction(self, X, pop_size):
        self.model.eval()                  
        with torch.no_grad():
            X_train = torch.FloatTensor(X).to(self.device)
            Y = self.model(X_train)
        Y = Y.data.squeeze().cpu().numpy()
        # Sort ascending
#            Y_sort = np.sort(Y, axis=None)  
        idx = np.argsort(Y)
#            X = X[idx[:pop_size]]          
        return idx[:pop_size]        

        
    def train(self, replay_buffer, Model, Model_backup, pop_size):
        self.model = Model.to(self.device)
        self.model_backup = Model_backup.to(self.device)
        self.model_optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
#        self.model_optimizer = Lamb(self.model.parameters(), lr=1e-3, weight_decay=0.01, betas=(.9, .999), adam=('lamb'))
#        self.model_optimizer = RAdam(self.model.parameters(), lr=1e-3)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer = self.model_optimizer, gamma = 0.95)
#        self.model.apply(init_weights)
#        self.model_backup.apply(init_weights)
        
        # sequential Sampling enable validation
        state, cost = replay_buffer.sequential_sample(len(replay_buffer.buffer))
        batch_size = 4
        while batch_size < ms.sqrt(len(replay_buffer.buffer)):
            batch_size *= 2

        best_valid_loss = np.inf
        early_stopping_tolerance = 0
        
        for t in range(self.max_training_iteration):
            self.model.train()
            if len(replay_buffer.buffer) - pop_size == 0:
                num_train = int(pop_size * 4/5.)
            else:
                num_train = len(replay_buffer.buffer) - pop_size
            # permutation
            batch_train = torch.randperm(num_train)
            batch_train = batch_train[(batch_train.numel() % batch_size):]
            batch_train = batch_train.view(-1, batch_size)
            running_loss = 0
            # self.model_backup.load_state_dict(self.model.state_dict())

            for i in range(batch_train.size(0)):        
                X_train = torch.FloatTensor(state[batch_train[i]]).to(self.device)
                Y_train = torch.FloatTensor(cost[batch_train[i]]).reshape((batch_size,1)).to(self.device)
                self.model_optimizer.zero_grad()
                pre_cost = self.model(X_train)
                loss = self.loss(pre_cost,Y_train)                       
                loss.backward()
                self.model_optimizer.step()    
                running_loss += loss.item()
                self.lr_scheduler.step()   

            # Validation                   
            self.model.eval()                  
            with torch.no_grad(): 
                X_val = torch.FloatTensor(state[num_train:]).to(self.device)
                Y_val = torch.FloatTensor(cost[num_train:]).reshape((-1,1)).to(self.device)
                Y_val_pre = self.model(X_val)
                new_valid_loss = self.loss(Y_val_pre,Y_val)
                new_valid_loss = new_valid_loss.item()
                best_valid_loss = min(new_valid_loss, best_valid_loss)
                if best_valid_loss == new_valid_loss:
                    early_stopping_tolerance = 0
                    self.model_backup.load_state_dict(self.model.state_dict())
                else:
                    early_stopping_tolerance += 1
                if early_stopping_tolerance >= 9:
                    self.model.load_state_dict(self.model_backup.state_dict())
                    Y_val_pre = self.model(X_val)
                    new_valid_loss = self.loss(Y_val_pre,Y_val)
                    new_valid_loss = new_valid_loss.item()
                    print('Final validation loss: {:.4f} '.format(new_valid_loss))
                    print('Training iteration : {}'.format(t))
                    break

                # if (new_valid_loss > pre_valid_loss) and (t >= 4):
                #     self.model.load_state_dict(self.model_backup.state_dict())
                #     Y_val_pre = self.model(X_val)
                #     new_valid_loss = self.loss(Y_val_pre,Y_val)
                #     new_valid_loss = new_valid_loss.item()
                #     early_stopping_tolerance += 1
                #     print('Final validation loss: {:.4f} '.format(new_valid_loss))
                #     break
                # else:                           
                #     pre_valid_loss = new_valid_loss

        print('Training loss: {:.4f} '.format(loss.item()))