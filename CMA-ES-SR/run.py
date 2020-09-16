# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:50:13 2020

@author: hongh
"""

from cost_function import Ackley, Branin, Schwefel, Sum_absolute, Buckin, Booth, Eggholder
from CMA_ES_SR import CMA_ES_SR
from CMA_ES_standard import CMA_ES
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    settings = {'nVar': 8 , 'MaxIt': 101}   
    Cost = Ackley
    n_trial = 5
    incumbent_CMA_ES = np.zeros((n_trial , settings['MaxIt']-1))
    incumbent_CMA_ES_SR = np.zeros((n_trial , settings['MaxIt']-1))
    num_proposed_solution = np.ones((n_trial , settings['MaxIt']-1))
    contri_model = np.zeros((n_trial , settings['MaxIt']-1))
    
    
    for i in range(n_trial):
        es = CMA_ES(**settings)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [Cost(x) for x in solutions])
        incumbent_CMA_ES[i] = np.squeeze(es.BestCost)     
    
    for i in range(n_trial):
        es = CMA_ES_SR(**settings)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [Cost(x) for x in solutions])
        incumbent_CMA_ES_SR[i] = np.squeeze(es.BestCost)  
        contri_model[i] = es.contri_model
        num_proposed_solution[i] = es.N_best_array

        
    ## Display Results   
    fig = plt.figure()    
    x = range(settings['MaxIt']-1)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    y = np.mean(incumbent_CMA_ES, axis = 0) 
    std = np.std(incumbent_CMA_ES, axis = 0)
    plt.plot(x,y)
    plt.fill_between(x, y-std, y+std ,color = 'r', alpha = 0.1, label = '95% CI')    
    y = np.mean(incumbent_CMA_ES_SR, axis = 0) 
    std = np.std(incumbent_CMA_ES_SR, axis = 0)
    plt.plot(x,y)
    plt.fill_between(x, y-std, y+std ,color = 'r', alpha = 0.1, label = '95% CI')
    # plt.ylim(0, 16)
    plt.legend(('Original CMA-ES', 'With Model'), loc='upper right')
    #grid on
    plt.show()
    #plt.gca().set_position([0, 0, 1, 1])
    fig.savefig("test.svg", format="svg")
      
    
    
    # plot number of samples from model
    fig = plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Number of proposed solutions')
    y = np.mean(num_proposed_solution, axis = 0) 
    std = np.std(num_proposed_solution, axis = 0)
    plt.plot(x,y)
    plt.fill_between(x, y-std, y+std , alpha = 0.1, label = '95% CI')
    plt.show()
    fig.savefig("test2.svg", format="svg")


    # plot how model contributes to the best
    fig = plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Best solution from model')
    y = np.mean(contri_model, axis = 0) 
    std = np.std(contri_model, axis = 0)
    plt.plot(x,y)
    plt.fill_between(x, y-std, y+std , alpha = 0.1, label = '95% CI')
    plt.show()
    fig.savefig("test3.svg", format="svg")
