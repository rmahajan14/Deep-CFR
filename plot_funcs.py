# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:48:35 2019

@author: ridhi
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
#
#path = 'DICT_Neural_Network_16_vs_128.pkl'
#with open(path, "rb") as pkl_file:
#    d = pickle.load(pkl_file)
####d = {
####     10:(1, 2),
####     20:(3, 5),
####     30:(10, 4),
####     40:(-4, 1),
####     50:(9, 2)
####}
#DIM1 = 16
#DIM2 = 128
#
#iteration = d.keys()
#result = [i[0] for i in d.values()]
#stdev = [i[1] for i in d.values()]
#plt.errorbar(iteration, result, stdev, linestyle='None', marker='^')
##plt.xticks(np.arange(min(iteration), max(iteration)+1, 10))
##plt.yticks(np.arange(min(result), max(result)+1, 10)) 
#plt.xlabel('Iteration')
#plt.ylabel('Game Result and Standard Deviation')
#plt.title(f'Neural Networks - Dimension {DIM1} vs {DIM2}')
#plt.savefig(f'Neural Networks - Dimension {DIM1} vs {DIM2}.png')
#plt.show()
#
#d = {
#     (10, 10):9,
#     (10, 20):1,
#     (10, 30):2,
#     (20, 10):6,
#     (20, 20):3,
#     (20, 30):6,
#     (30, 10):6,
#     (30, 20):8,
#     (30, 30):5,
#     }
DIM = 128
for DIM in [16, 64, 128]:
    path = f'DICT_Iterations_agents_{DIM}.pkl'
    with open(path, "rb") as pkl_file:
        d = pickle.load(pkl_file)
    
    x_list = []
    y_list = []
    value_list = []
    
    
    for item in d.items():
        x = item[0][0]
        y = item[0][1]
        value = item[1]
        
        x_list += [10*x]
        y_list += [y*10]
        value_list += [value]
    
    
    plt.figure(figsize=(10, 10))
    #fig, ax = plt.subplots()
    plt.scatter(x_list, y_list, c=value_list, 
                s=200,
    #            cmap='YlOrBr',
    #            cmap='binary'
                cmap='seismic'
                )
    plt.plot([0,90], [0,90], color='black')
    #plt.xticks(np.arange(min(x_list), max(x_list*10)+1, 10))
    #plt.yticks(np.arange(min(y_list), max(y_list*10)+1, 10))    
    #ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    plt.xlabel('Agent 1 Iterations', fontsize=15)
    plt.ylabel('Agent 2 Iterations', fontsize=15)
    plt.title(f'Agent vs. Agent - NN Dimension {DIM}', fontsize=20)
    plt.colorbar()
    plt.savefig(f'Differently Trained Agents {DIM}.png')
    plt.show()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
