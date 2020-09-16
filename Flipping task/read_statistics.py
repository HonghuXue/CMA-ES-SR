# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:10:29 2019

@author: Honghu Xue
"""

import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as pl


#df = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])
#for i in range(6):
#    df.loc[i] = ['name' + str(i)] + list(np.random.randint(10, size=2))
#print(len(df))



# plot the fitness_function

read_training_data = True

if read_training_data == True:
    x, y = [], []
    df = pickle.load(open('statistics.pkl', 'rb'))
    for i in range(len(df)):
        x.append(np.mean(df.loc[i]['fitness_value']))
        y.append(np.std(df.loc[i]['fitness_value']))
    a = np.linspace(0, len(df), len(df))
    b = np.array(x)
    std = np.array(y)
    print(len(b))
    #b += np.random.normal(0, 0.1, size=y.shape)
    pl.plot(a, b, color = 'b', alpha = 0.6, label='pop = 50')
    pl.fill_between(a, b-std, b+std ,color = 'b', alpha = 0.1, label = '95% CI')  ##539caf                          
    pl.title('Fitness function average')
    pl.xlabel('Iterations')
    pl.ylabel('fitness value')  
#    pl.ylim([0,700])
    pl.show()
    
     
    # plot the incumbent setting
    x =[]
    for i in range(len(df)):
        x.append(np.min(df.loc[i]['fitness_value']))
    a = np.linspace(0, len(df), len(df))
    b = np.array(x)
    #b += np.random.normal(0, 0.1, size=y.shape)
    pl.plot(a, b, 'b', alpha= 0.6, label='pop = 50')
    pl.title('Incumbent settings')
    pl.xlabel('Iterations')
    pl.ylabel('fitness value')  
    pl.show()
    
     
    # plot Sigma
    x =[]
    for i in range(len(df)):
        x.append(np.min(df.loc[i]['sigma']))
    a = np.linspace(0, len(df), len(df))
    b = np.array(x)
    #b += np.random.normal(0, 0.1, size=y.shape)
    pl.plot(a, b,  'b', alpha= 0.6)
    pl.title('Std_deviation')
    pl.xlabel('Iterations')
    pl.ylabel('Std_dev')  
    pl.show()


    # plot episodic spillage
    x, y = [], []
    for i in range(len(df)):
        x.append(np.mean(df.loc[i]['episodic_spillage']))
        y.append(np.std(df.loc[i]['episodic_spillage']))
    a = np.linspace(0, len(df), len(df))
    b = np.array(x)
    std = np.array(y)
    pl.plot(a, b, color = 'b', alpha = 0.6, label='pop = 50')
    pl.fill_between(a, b-std, b+std ,color = 'b', alpha = 0.1, label = '95% CI')  ##539caf                          
    pl.title('Episodic spillage average')
    pl.xlabel('Iterations')
    pl.ylabel('Spillage')  
    pl.show()


    # plot max rotated angle
    x, y = [], []
    for i in range(len(df)):
        x.append(np.mean(df.loc[i]['max_rotated_angle']))
        y.append(np.std(df.loc[i]['max_rotated_angle']))
    a = np.linspace(0, len(df), len(df))
    b = np.array(x)
    std = np.array(y)
    pl.plot(a, b, color = 'b', alpha = 0.6, label='pop = 50')
    pl.fill_between(a, b-std, b+std ,color = 'b', alpha = 0.1, label = '95% CI')  ##539caf                          
    pl.title('Max rotated angle')
    pl.xlabel('Iterations')
    pl.ylabel('Max rotated angle [deg]')  
    pl.show()
    
    
    # plot episodic spillage
    x, y = [], []
    for i in range(len(df)):
        x.append(np.mean(df.loc[i]['final_angle']))
        y.append(np.std(df.loc[i]['final_angle']))
    a = np.linspace(0, len(df), len(df))
    b = np.array(x)
    std = np.array(y)
    pl.plot(a, b, color = 'b', alpha = 0.6, label='pop = 50')
    pl.fill_between(a, b-std, b+std ,color = 'b', alpha = 0.1, label = '95% CI')  ##539caf                          
    pl.title('Final angle')
    pl.xlabel('Iterations')
    pl.ylabel('Final angle [deg]')  
    pl.show()

    # plot step spillage ----generation example
#    df = pd.DataFrame(columns = ['step_spillage'])
#    for iter in range(5):# total iterations
#        a = []
#        for i in range(120): # 120 offsprings        
#            a.append(np.random.randint(100, size = 500))
#        df.loc[len(df)] = [a] # can't use np.array(a)
#    
#    print(df)
#    for i in range(len(df)):
#        if i%5 == 0:
#            b = np.mean(np.array(df.loc[i]['step_spillage']),axis = 0)
#            std =  np.std(np.array(df.loc[i]['step_spillage']),axis = 0)
#            a = np.linspace(1, 500, 500)
#            pl.plot(a, b, color = 'b', alpha = 0.6, label='pop = 50')
#            pl.fill_between(a, b-std, b+std ,color = 'b', alpha = 0.1, label = '95% CI')  ##539caf                          
#            pl.title('Fitness function average')
#            pl.xlabel('Iterations')
#            pl.ylabel('fitness value')  
#            pl.show()
    
    # plot step_spillage
#    df = pickle.load(open('statistics_step_spillage.pkl', 'rb'))
#    print(df)
#    for i in range(len(df)):
#        b = df.loc[10*(i+1)]['step_spillage']
#        b = np.mean(np.array(df.loc[10*(i+1)]['step_spillage']),axis = 0)
#        std =  np.std(np.array(df.loc[10*(i+1)]['step_spillage']),axis = 0)
#        a = np.linspace(1, 501, 501)
#    #    #b += np.random.normal(0, 0.1, size=y.shape)
#        pl.plot(a, b, color = 'b', alpha = 0.6, label='pop = 50')
#        pl.fill_between(a, b-std, b+std ,color = 'b', alpha = 0.1, label = '95% CI')  ##539caf                          
#        pl.title('Spillage w.r.t. time %s-th iteration' %(10*(i+1)))
#        pl.xlabel('Time [10ms]')
#        pl.ylabel('Spillage')  
#        pl.show()
    

else:
    mean, std = [], []
    mean2, std2 = [], []
    df1 = pickle.load(open('validation_spillage.pkl', 'rb'))
    df2 = pickle.load(open('validation_distance.pkl', 'rb'))
    print(df2)
    listOfColumnNames = list(df1.columns.values)
    x = np.arange(len(listOfColumnNames))
    for i in listOfColumnNames:
        mean.append(df1[i].mean())
        std.append(df1[i].std())
        mean2.append(df2[i].mean())
        std2.append(df2[i].std())
    fig, ax = pl.subplots(figsize=(16, 12))
    ax.bar(x, mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Number of spilled particles')
    ax.set_xticks(x)
    ax.set_xticklabels(listOfColumnNames)
    ax.set_title('Validation of DMPs on different target poses')
    ax.yaxis.grid(True)
    
    # Save the figure and show
    pl.tight_layout()
    pl.savefig('validation_spillage.png')
    pl.show()
    
    fig, ax = pl.subplots(figsize=(16, 12))
    ax.bar(x, mean2, yerr=std2, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Distance difference [m]')
    ax.set_xticks(x)
    ax.set_xticklabels(listOfColumnNames)
    ax.set_title('Validation of DMPs on different target poses')
    ax.yaxis.grid(True)    
    pl.tight_layout()
    pl.savefig('validation_distance.png')
    pl.show()
    
    
    
#    aluminum = np.array([6.4e-5 , 3.01e-5 , 2.36e-5, 3.0e-5, 7.0e-5, 4.5e-5, 3.8e-5, 4.2e-5, 2.62e-5, 3.6e-5])
#    copper = np.array([4.5e-5 , 1.97e-5 , 1.6e-5, 1.97e-5, 4.0e-5, 2.4e-5, 1.9e-5, 2.41e-5 , 1.85e-5, 3.3e-5 ])
#    steel = np.array([3.3e-5 , 1.2e-5 , 0.9e-5, 1.2e-5, 1.3e-5, 1.6e-5, 1.4e-5, 1.58e-5, 1.32e-5 , 2.1e-5])
#    
#    aluminum_mean = np.mean(aluminum)
#    copper_mean = np.mean(copper)
#    steel_mean = np.mean(steel)   
#    aluminum_std = np.std(aluminum)
#    copper_std = np.std(copper)
#    steel_std = np.std(steel)
#    # Create lists for the plot
#    materials = ['Aluminum', 'Copper', 'Steel']
#    x_pos = np.arange(len(materials))
#    CTEs = [aluminum_mean, copper_mean, steel_mean]
#    error = [aluminum_std, copper_std, steel_std]
#    # Build the plot
#    fig, ax = pl.subplots()
#    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
#    ax.set_ylabel('Coefficient of Thermal Expansion ($\degree C^{-1}$)')
#    ax.set_xticks(x_pos)
#    ax.set_xticklabels(materials)
#    ax.set_title('Validation of DMPs on different target poses')
#    ax.yaxis.grid(True)
#    
#    # Save the figure and show
#    pl.tight_layout()
#    pl.savefig('bar_plot_with_error_bars.png')
#    pl.show()
#        
#



