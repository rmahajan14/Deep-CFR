# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:17:35 2019

@author: ridhi
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel(r'HUNL_EXAMPLE SINGLE_stack_20000_ LBR Total,tag_Evaluation_MBB_per_G.xlsx')


df = df.loc[df['Step'] <= 80]

x = df['Step']
y_val = df['Value LBR']
y6 = df['Param = .6']
y7 = df['Param = .7']
y8 = df['Param = .8']
y85 = df['Param = 0.85']
y9 = df['Param=.9']

plt.figure(figsize=(10, 10))
#plt.plot(x, y_val)
#plt.plot(x, y6, label='.6')
#plt.plot(x, y7, label='.7')
#plt.plot(x, y8, label='.8')
plt.scatter(x, y85, label='.85')
#plt.legend()
plt.plot(x, y85)
#plt.plot(x, y9, label='.9')

plt.ylabel('Exploitability (mbb/g)', fontsize=15)
plt.xlabel('Iteration', fontsize=15)

plt.title('Exploitability vs Iterations: training for HUNL', 
          fontsize=20)

plt.savefig('HUNL_results_final.png')
plt.show()


