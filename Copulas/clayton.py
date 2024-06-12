# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:30:26 2023

@author: Julian
"""
#%% Import packages

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

#%% Clayton copula

#function cumulative

def Clayton(XY, alpha, diff = False):
    
    if diff:
            
        x = XY.x.values.ravel()
        y = XY.y.values.ravel().reshape(-1,1)
        
        return (x**(-alpha)+y**(-alpha)-1)**(-1/alpha)
    
    else:
        return (XY.x**(-alpha)+XY.y**(-alpha)-1)**(-1/alpha)

#generator

def inv_Clayton(u, c, alpha):
    
    return ((c**(-alpha/(1+alpha))-1)*u**(-alpha)+1)**(-1/alpha)

# =============================================================================
# =============================================================================
#  #%%
#  
# alpha = 8
#  
# u = ss.uniform(0,1).rvs(1000)
#  
# c = ss.uniform(0,1).rvs(1000)
#  
# v = inv_clayton(u, c, alpha)
#  
# plt.scatter(u,v)
#  
#  #%%
#  
# XY = pd.DataFrame([u, v], index = ['x','y']).T
#  
# print(clayton(XY,alpha))
#  
#  
#  #%%
#  
# from tools import Qn
# from tools import Q
# from tools import Cn
#  
#  #%%
#  
# q = Q(XY.x, XY.y, clayton, [alpha])
#  
# qn = Qn(XY, XY.x, XY.y, Cn).apply(lambda x: np.nan if abs(x)>1 else x)
#  
#  #%%
#  
# T_2 = ((qn-q)**2).sum()
#  
# print(T_2)
#  
# =============================================================================
# #%%
# print(ss.chi2(1).sf(T_2))
# =============================================================================
