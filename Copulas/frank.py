# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 23:21:53 2024

@author: Julian
"""

import scipy.stats as ss 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% Frank Copula

def Frank(XY, alpha, diff = False):
    
    if diff:
        
        x = XY.x.values.ravel()
        y = XY.y.values.ravel().reshape(-1,1)
        
        return -(1/alpha)*np.log(1 + ((np.exp(-alpha*x)-1)*
                                  (np.exp(-alpha*y)-1)/
                                  (np.exp(-alpha)-1)))
        
    else:
    
        return -(1/alpha)*np.log(1 + ((np.exp(-alpha*XY.x)-1)*
                                  (np.exp(-alpha*XY.y)-1)/
                                  (np.exp(-alpha)-1)))

def inv_Frank(u, c, alpha):
    
    return -(1/alpha)*np.log(1+(c*(np.exp(-alpha)-1)/(
        
        np.exp(-alpha*u)*(1-c)+c
        
        )))

# =============================================================================
# #%% Tries
# 
# u = ss.uniform(0,1).rvs(1000)
# c = ss.uniform(0,1).rvs(1000)
# alpha = 30
# v = inv_Frank(u, c, alpha)
# #%%
# 
# plt.scatter(u, v, alpha = 0.5, s=10)
# 
# #%%
# 
# XY = pd.DataFrame([u,v], index = ["x","y"]).T
# 
# print(Frank(XY,alpha).reset_index().sort_values(by=0))
# =============================================================================
