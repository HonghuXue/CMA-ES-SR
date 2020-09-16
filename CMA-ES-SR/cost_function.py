# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:11:46 2020

@author: hongh
"""

import numpy as np



#@Ackley 
def Ackley(x):
    z=20*(1-np.exp(-0.2*np.sqrt(np.mean(np.square(x)))))+np.exp(1)-np.exp(np.mean(np.cos(2*np.pi*x))) 
    return z

#@ eggholder , 2dimensional  , with local optimum 
def Eggholder(x):
    x2 = x[1]
    x1 = x[0]
    term1 = -(x2+47) * np.sin(np.sqrt(np.abs(x2+x1/2+47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1-(x2+47))))
    return term1 + term2

# Buckin  local optimum , 2 dim
def Buckin(x):
    term1 = 100 * np.sqrt(np.abs(x[1] - 0.01*x[0]**2))
    term2 = 0.01 * np.abs(x[0]+10)
    return term1+ term2

# Colville function , 4 dim input , other function
def Colville(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    term1 = 100 * (x1**2-x2)**2
    term2 = (x1-1)**2
    term3 = (x3-1)**2
    term4 = 90 * (x3**2-x4)**2
    term5 = 10.1 * ((x2-1)**2 + (x4-1)**2)
    term6 = 19.8*(x2-1)*(x4-1)
    return term1 + term2 + term3 + term4 + term5 + term6


# EASOM 2 dimensional, a sudden drop
def EASOM(x):
    y1 = -np.cos(x[0])*np.cos(x[1])
#    y2 = np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)
    y2 = np.power(1.01, -(x[0] - np.pi)**2 - (x[1] - np.pi)**2)
    return y1*y2


#@Schwefel
def Schwefel(x):
    z = np.sum(np.multiply(-x,np.sin(np.sqrt(np.abs(x))))) + 418.9828878*len(x)
    return z

#@Michalewicz   
def Michalewicz(x):  # mich.m
    n = len(x)
    j = np.arange( 1., n+1 )
    return - np.sum( np.sin(x) * np.sin( j * x**2 / np.pi ) ** (2 * .5) )

#@rosenbrock
def Rosenbrock(x):
    x0 = x[:-1]
    x1 = x[1:]
    return (sum( (1 - x0) **2 ) + 100 * sum( (x1 - x0**2) **2 ))


#rastrigin  ---- good example
def Rastrigin(x):  
    n = len(x)
    return 10*n + np.sum( x**2 - 10 * np.cos( 2 * np.pi * x ))

#griewank ---- good example
def Griewank(x, fr=4000):
    n = len(x)
    j = np.arange( 1., n+1 )
    s = np.sum( x**2 )
    p = np.prod( np.cos( x / np.sqrt(j) ))
    return s/fr - p + 1


# Branin   , must be 2 dimensional
def Branin(x):
    b = (5.1 / (4.*np.pi**2))
    c = (5. / np.pi)
    t = (1. / (8.*np.pi))
    return 1.*(x[1]-b*x[0]**2+c*x[0]-6.)**2+10.*(1-t)*np.cos(x[0])+10.

# Booth function, plane must be 2 dimensional
def Booth(x):
    return (x[0] + 2*x[1] -7)**2 + (2*x[0]+x[1]-5)**2


# Self-defined function
def Sum_absolute(x):
    return np.sum(np.abs(x)) + 5.


#SIX-HUMP CAMEL FUNCTION valley-based
def Six_hump(x):  
    term1 = (4-2.1*x[0]**2+(x[0]**4)/3) * x[0]**2
    term2 = x[0]*x[1]
    term3 = (-4+4*x[1]**2) * x[1]**2           
    return term1 + term2 + term3

#ZAKHAROV FUNCTION   -- plane
def ZAKHAROV(x):
    d = len(x)
    sum1 = 0
    sum2 = 0
    for i in range(d):
    	xi = x[i]
    	sum1 = sum1 + xi**2
    	sum2 = sum2 + 0.5*(i+1)*xi
    return sum1 + sum2**2 + sum2**4