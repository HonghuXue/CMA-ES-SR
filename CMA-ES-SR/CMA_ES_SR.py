# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:14:02 2020

@author: hongh
"""

import numpy as np
import random
import math as ms
from cost_function import Ackley, Branin, Schwefel, Sum_absolute, Buckin, Booth, Eggholder
from utils import ReplayBuffer, init_weights
from model import Model
from model_train import Model_learn

import torch


class CMA_ES_SR(object):
    def __init__(self, **kwargs):
        self.nVar = kwargs.get('nVar',4)
        self.VarMin= kwargs.get('VarMin',-10)
        self.VarMax= kwargs.get('VarMin',10)
        assert self.VarMin < self.VarMax, 'Varmin > Varmax'
        self.VarSize = [1, self.nVar]    
#        self.seed = kwargs.get('seed',np.random.randint(1,999999)) 
#        np.random.seed(self.seed)
#        random.seed(self.seed)
        # Maximum Number of Iterations
        self.MaxIt=kwargs.get('MaxIt',200)
        # Initial variance
        self.sigma0 = kwargs.get('sigma0' ,0.3*(self.VarMax-self.VarMin))
        self.pop_size = kwargs.get('pop_size',4+ms.ceil(3*ms.log(self.nVar)))

        self.stopfitness = kwargs.get('target_value',-np.inf)
        # Number of Parents
        self.mu=round(self.pop_size/2)        
        # Parent Weights
        self.w=np.log(self.mu+0.5)-np.log(range(1,self.mu+1))
        self.w=self.w/sum(self.w)       
        # Number of Effective Solutions
        self.mu_eff=sum(self.w)**2/sum(np.power(self.w,2))   #original: 1./sum(np.power(w,2))        
        # Step Size Control Parameters (c_sigma and d_sigma)        
        self.cs = (self.mu_eff+2)/(self.nVar+self.mu_eff+5)
        self.ds = 1 + self.cs + 2 * max(np.sqrt((self.mu_eff-1)/(self.nVar+1))-1, 0)
        self.ENN = np.sqrt(self.nVar)*(1-1./(4*self.nVar)+1./(21*self.nVar**2))  # chiN       
        # Covariance Update Parameters
        self.cc=(4.+self.mu_eff/self.nVar)/(4+self.nVar+2*self.mu_eff/self.nVar)
        self.c1=2./((self.nVar+1.3)**2+self.mu_eff)
        self.alpha_mu=2
        self.cmu=min(1-self.c1,self.alpha_mu*(self.mu_eff-2+1./self.mu_eff)/((self.nVar+2)**2+self.alpha_mu*self.mu_eff/2.))
        self.hth=(1.4+2/(self.nVar+1))*self.ENN                
        self.sigma = np.zeros(self.MaxIt)
        self.sigma[0]=self.sigma0
        # Initialization
#        self.ps=np.zeros(MaxIt, dtype=object)   #evolution paths for sigma
#        self.pc=np.zeros(MaxIt, dtype=object)   #evolution paths for C
#        self.C=np.zeros(MaxIt, dtype=object)  
        self.ps=np.zeros(self.VarSize)
        self.pc=np.zeros(self.VarSize)
        self.C=np.eye(self.nVar)
        
        #https://stackoverflow.com/questions/14922586/matlab-struct-array-to-python
        self.M = np.zeros(self.MaxIt, dtype = [('Position',np.ndarray),('Step',np.ndarray),('Cost',np.ndarray)])       
        self.M[0]['Position'] = kwargs.get('initial_position',np.random.uniform(self.VarMin,self.VarMax,self.nVar))  
        self.M[0]['Step']=np.zeros(self.VarSize)
        self.M[0]['Cost'] = np.inf
        self.BestSol=self.M[0]
        self.BestCost=np.zeros((self.MaxIt-1,1))
        self.itr = 0
        self.replay_buffer = ReplayBuffer()
        
        # Model Initialization
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.nVar).to(self.device)
        self.model.apply(init_weights)
        self.model_backup = Model(self.nVar).to(self.device)
        self.model_learn = Model_learn(self.model ,self.device)
        self.N_best_model_samples = 2 # initial number of samples mutated by the model
        self.N_max = kwargs.get('N_max' ,4096)
        self.N_model_solution = min(self.N_max ,self.pop_size**self.nVar)
        self.contri_model = np.zeros(self.MaxIt-1)
        self.N_best_array = np.ones(self.MaxIt-1)
        print('Initialization finished')


    def ask(self):
        ''' Returning the solution in the form of list '''
        pop_tmp=np.zeros(self.N_model_solution, dtype = [('Position',np.ndarray),('Step',np.ndarray),('Cost',np.ndarray)])
        for i in range(self.N_model_solution):                   
            pop_tmp[i]['Step']=np.random.multivariate_normal(np.zeros(self.nVar),self.C).T
            pop_tmp[i]['Position']=self.M[self.itr]['Position'] + self.sigma[self.itr]*pop_tmp[i]['Step']
        
        self.model_backup.load_state_dict(self.model.state_dict())
        model_sol_idx = self.model_learn.model_prediction(np.array(list(pop_tmp['Position']), dtype=np.float), self.pop_size)           

        # Selection...
        self.pop=np.zeros(self.pop_size, dtype = [('Position',np.ndarray),('Step',np.ndarray),('Cost',np.ndarray)])
        used_model_idx = []
        # from the model
        print('Number of samples from model : {}'.format(self.N_best_model_samples))
        for i in range(self.N_best_model_samples):
            self.pop[i]['Position'] = pop_tmp[model_sol_idx[i]]['Position']
            self.pop[i]['Step'] = pop_tmp[model_sol_idx[i]]['Step']
            used_model_idx.append(model_sol_idx[i])
        # not from the model
        for i in range(self.N_best_model_samples, self.pop_size):
            j = i# to avoid the repetition :
            while j in used_model_idx:
                j = np.random.randint(low= 0, high=self.N_model_solution - 1)
            self.pop[i]['Position'] = pop_tmp[j]['Position']
            self.pop[i]['Step'] = pop_tmp[j]['Step']

        # Adding the mean
#        self.pop[self.pop_size]['Step']=np.zeros(self.nVar).T
#        self.pop[self.pop_size]['Position'] = self.M[self.itr]['Position']  
        
        self.indexes = np.array(range(self.N_best_model_samples))
        
        return self.pop['Position'].tolist()


    def stop(self):
        if self.itr >= self.MaxIt-1 or self.BestCost[self.itr]<=self.stopfitness or np.mean(self.sigma)<= 1e-8:
            return True
        else:
            return False

    
    def tell(self, solutions, answers): # solutions and answers are lists
        for i in range(len(answers)):
            self.pop[i]['Cost']=answers[i]
            self.replay_buffer.add((self.pop[i]['Position'],self.pop[i]['Cost']))
            # Update Best Solution Ever Found
            if self.pop[i]['Cost'] < self.BestSol['Cost']:
                self.BestSol=self.pop[i]
        
        # Update model and relevant parameters
        self.model_learn.train(self.replay_buffer, self.model, self.model_backup, self.pop_size)#self.pop_size+1
        # Compute np.std & np.mean , update  num_sample_from_model
        proposed_solution_mean = np.mean(self.pop['Cost'][self.indexes])
        proposed_solution_std = np.std(self.pop['Cost'][self.indexes])
#        best_model_sol = np.min(self.pop['Cost'][self.indexes])
        original_index = set(range(len(self.pop['Cost'])))
        model_index = set(self.indexes)
        cov_index = np.array(list(original_index - model_index))  
        covariance_solution_mean = np.mean(self.pop['Cost'][cov_index])
        covariance_solution_std = np.std(self.pop['Cost'][cov_index])
#        best_covariance_sol = np.min(self.pop['Cost'][cov_index])
        print('proposed mean : {}'.format(proposed_solution_mean))
        print('covariance mean : {}'.format(covariance_solution_mean))
        self.N_best_array[self.itr] = self.N_best_model_samples            
        if covariance_solution_mean - covariance_solution_std > proposed_solution_mean - proposed_solution_std:
            self.N_best_model_samples += 1
        else:
            self.N_best_model_samples -= 1       
        # if best_model_sol < best_covariance_sol:
        #     num_sample_from_model += 1
        # else:
        #     num_sample_from_model -= 1
        self.N_best_model_samples = max(2, min(self.N_best_model_samples, self.mu, self.pop_size-2))
           
        # Sort Population
        Costs=self.pop['Cost']
        SortOrder = np.argsort(Costs)
        Costs = np.sort(Costs)
        self.pop=self.pop[SortOrder]        
        # Save Results
        self.BestCost[self.itr]=self.BestSol['Cost']
        
        # Display Results
        print('Iteration : {} , Best Cost = {}'.format(self.itr, self.BestCost[self.itr]))
        print('Sigma : {}'.format(np.mean(self.sigma[self.itr])))        
                    
        # Update Mean
        self.M[self.itr+1]['Step']=0
        for j in range(self.mu):
            self.M[self.itr+1]['Step']=self.M[self.itr+1]['Step']+self.w[j]*self.pop[j]['Step'] #w[j] is a scalar           
            # Not sure ...Add contribution to the model
            if np.any(self.indexes == SortOrder[j]):
                self.contri_model[self.itr] += 1
                
        self.M[self.itr+1]['Position']=self.M[self.itr]['Position']+self.sigma[self.itr]*self.M[self.itr+1]['Step']  #sigma[g] is a scalar
         
        # Update Step Size
        self.ps=(1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mu_eff) * np.linalg.lstsq(np.linalg.cholesky(self.C).T, self.M[self.itr+1]['Step'])[0].T  #np.matmul(M[g+1]['Step'], np.linalg.inv(np.linalg.cholesky(C[g])))
        self.sigma[self.itr+1]=self.sigma[self.itr]*np.power(np.exp(self.cs/self.ds*(np.linalg.norm(self.ps)/self.ENN-1)),0.3)
    
        # Update Covariance Matrix
        if np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*(self.itr+1))) < self.hth:# Original np.linalg.norm(ps[g+1])/np.sqrt(1- (1-cs)**(2*(g+1))) < hth:
            self.hs=1
        else:
            self.hs=0    
        self.delta=(1-self.hs)*self.cc*(2-self.cc)
        self.pc=(1-self.cc)*self.pc + self.hs*np.sqrt(self.cc*(2-self.cc)*self.mu_eff)*self.M[self.itr+1]['Step']
        self.C=(1-self.c1-self.cmu)*self.C + self.c1*(np.matmul(self.pc.T,self.pc) + self.delta*self.C)
        for j in range(self.mu):
            self.C=self.C + self.cmu*self.w[j]*np.matmul(self.pop[j]['Step'].reshape(1,-1).T,self.pop[j]['Step'].reshape(1,-1))
        
        # If Covariance Matrix is not Positive Defenite or Near Singular
        E, V = np.linalg.eig(self.C)  
        if any(i<0 for i in E):
            print('not positive')
            E[E<0]=0 #max(E,0)        
            self.C= np.matmul(np.matmul(V,E) , np.linalg.inv(V))
            print(np.all(np.linalg.eigvals(self.C) > 0))
            print('ENFORCING Semi-positive definite')             
        self.itr += 1    


    def save_dataset(self):
        '''Save [(candidate, cost)]
        https://github.com/kwikteam/npy-matlab
        https://towardsdatascience.com/why-you-should-start-using-npy-file-more-often-df2a13cc0161
        '''
        state, cost = self.replay_buffer.sequential_sample(self.replay_buffer.size)
        # save to npy file
        np.save('state.npy', state)
        np.save('cost.npy', cost)

 
    

          
if __name__ == '__main__':
    settings = {}
    es = CMA_ES_SR()
    Cost = Ackley
    count = 0
    while es.stop():
        solutions = es.ask()
        es.tell(solutions, [Cost(x) for x in solutions])
    print(es.BestSol)
    print(len(es.BestCost))
    es.save_dataset()