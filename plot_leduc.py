# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:17:35 2019

@author: ridhi
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'Leduc 64.csv')
df2 = pd.read_csv(r'Leduc 128.csv')
df3 = pd.read_csv(r'Leduc16.csv')



df = df.loc[df['Step'] <= 100]
df2 = df.loc[df['Step'] <= 100]
df3 = df.loc[df['Step'] <= 100]

x = df['Step']
y_val = df['Value']


x2 = df2['Step']
y_val2 = df2['Value']

x3 = df3['Step']
y_val3 = df3['Value']


plt.figure(figsize=(10, 10))
plt.scatter(x, y_val)
#plt.legend(fontsize=15)
plt.plot(x, y_val)
plt.ylabel('Exploitability (ma/g)', fontsize=15)
plt.xlabel('Iteration', fontsize=15)
plt.title('Exploitability vs Iterations: Dimension 64', 
          fontsize=20)
plt.savefig('Leduc_64.png')
plt.show()


plt.figure(figsize=(10, 10))
plt.scatter(x2, y_val2)
#plt.legend(fontsize=15)
plt.plot(x2, y_val2)
plt.ylabel('Exploitability (ma/g)', fontsize=15)
plt.xlabel('Iteration', fontsize=15)
plt.title('Exploitability vs Iterations: Dimension 128', 
          fontsize=20)
plt.savefig('Leduc_128.png')
plt.show()


plt.figure(figsize=(10, 10))
plt.scatter(x3, y_val3)
#plt.legend(fontsize=15)
plt.plot(x3, y_val3)
plt.ylabel('Exploitability (ma/g)', fontsize=15)
plt.xlabel('Iteration', fontsize=15)
plt.title('Exploitability vs Iterations: Dimension 16', 
          fontsize=20)
plt.savefig('Leduc_16.png')
plt.show()