# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 00:39:34 2024

@author: Julian
"""

#%% Import packages

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

#%% Gumbel-Hougard copula

def Gumbel_Hougaard(XY, alpha, diff = False):
    
    if diff:
            
        x = XY.x.values.ravel()
        y = XY.y.values.ravel().reshape(-1,1)
            
        return np.exp(-((
                -np.log(x))**alpha + (
                    -np.log(y))**alpha)**(1/alpha))
    
    else:
        
        return np.exp(-((-np.log(XY.x))**alpha + (-np.log(XY.y))**alpha)**(1/alpha))

def inv_Gumbel_Hougaard(v1, v2, x, alpha):
    
    # Se uso la funcion generadora de la forma ui = phi(Ei/M) donde Ei es una exponencial y M es la funcion de distribucion caracteristica asociada a la copula
    # y phi es la funcion generadora de la copula arquimediana
    return np.exp(-(-np.log(v1)/x)**(1/alpha)), np.exp(-(-np.log(v2)/x)**(1/alpha))

#%% Try
# =============================================================================
# 
# alpha = 5
# 
# n = 1000
# 
# x = ss.levy_stable(1/alpha,
#                    1,
#                    alpha==1,# Si ocurre entonces 1, caso contrario 0
#             np.cos(np.pi/(2*alpha))**alpha).rvs(n)
# #Siempre esta en S1
# 
# v1 = ss.uniform(0,1).rvs(n)
# 
# v2 = ss.uniform(0,1).rvs(n)
# 
# u, v = inv_Gumbel_Hougaard(*[v1,v2], x, alpha)
# #%%
# 
# plt.scatter(u,v, marker='.', alpha=0.7)
# 
# #%%
#   
# XY = pd.DataFrame([u, v], index = ['x','y']).T
#   
# print(Gumbel_Hougaard(XY,alpha))
#   
#   
#   #%%
#   
# from tools import Qn
# from tools import Q
# from tools import Cn
# from tools import Dn
# from Copulas.clayton import Clayton
#   
#   #%%
# q = Q(XY.x, XY.y, Gumbel_Hougaard, [alpha])  
# q_ = Q(XY.x, XY.y, Clayton, [alpha])
#   
# qn = Qn(XY, XY.x, XY.y, Cn).apply(lambda x: np.nan if abs(x)>1 else x)
#   
#   #%%
#   
# T_2 = ((qn-q)**2).sum()
# T_2_ = ((qn-q_)**2).sum()
# print(T_2)
# print(T_2_)
#   
# #=============================================================================
# #%%
# 
# #=============================================================================
# 
# =============================================================================
