# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:46:00 2024

@author: Julian
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

#%% Gumbel-Barnett copula

def Gumbel_Barnett(XY, alpha):
    
    return XY.x+XY.y-1+(1-XY.x)*(1-XY.y)*np.exp(
        -alpha*np.log(1-XY.x)*np.log(1-XY.y)
        )

def Gumbel_Barnett_2(UV, alpha):
    
    x = -np.log(1-UV.u)
    y = -np.log(1-UV.v)
    
    return 1 - np.exp(-x) - np.exp(-y) + np.exp(-y-x-alpha*x*y)
    

def inv_Gumbel_Barnett(u1, c, alpha):
    
    import scipy.optimize as so
    
    y = -np.log(1-u1)
    
    #f = lambda x, alpha, y, c: (-np.exp(-x*(1+alpha*y))*(alpha*x+1)-c)
    #f = lambda x, alpha, y, c: (c-np.exp(-y)-np.exp(-x-y-alpha*x*y)*(-1-alpha*x))
    f = lambda x, alpha, y, c: 1-(1+alpha*x)*np.exp(-(1+alpha*y)*x)-c
    
    x = []
    
    for i,j in zip(y,c):
        
        try:
        
            x.append(so.root_scalar(f, args=(alpha, i, j), bracket = (0, 20)).root)
     
        except:
            
            x.append(so.root_scalar(f, args=(alpha, i, j), bracket = (-20, 0)).root)
     
    x = np.array(x)
    
    f = 1-np.exp(-abs(x))
    
    #f = x.ravel()
    
    #f[x>=0] = f2 ; f[x<0] = f1
    
    return f
    
    #----- Codigo antiguo
    
    #print(y)
    #print(c)
    
    #
    
    #f = lambda x, alpha, y, c: (np.exp(-x-y-alpha*y*x)*(-1-alpha*x)-c)
    #f = lambda x: np.exp(-x-y-alpha*y*x)*(-1-alpha*x)-c
    #f = lambda x, alpha, y, c: 1-np.exp(-alpha*np.log(1-x)*np.log(1-y))*(1-x)*(1-alpha*np.log(1-x))-c
    
     
   
    #def f(x, alpha, y, c):
        
     #   F = -np.exp(y)+np.exp(y+x-alpha*x*y)*(1 - alpha*x) - c
        
      #  return F
            
            
    
    #s = so.root(f, [0]*len(u1), method = 'hybr' , args=(alpha, y, c))
   
    #x = s['x']
    
    #s = so.minimize(f, [0]*len(u1), method = "L-BFGS-B",args = (alpha, y, c),
                   # bounds=((0, float("inf")) for i in range(len(u1))))
    
    #x = s[0]
    
    #print(x)

    #x = so.fsolve(f, [0]*len(u1), args=(alpha, y, c))
    
    #print(x)
    
    #return 1-np.exp(-x)

#%% tries
# =============================================================================
# 
# u1 = ss.uniform(0, 1).rvs(1000)
# 
# c = ss.uniform(0, 1).rvs(1000)
# 
# alpha = 0.2
# 
# u, v = inv_Gumbel_Barnett(u1, c, alpha)
# 
# plt.scatter(u, v, alpha = 0.8, s = 8)
# =============================================================================

#%% T tries

# =============================================================================
# from tools import Qn
# from tools import Cn
# from tools import Dn
# from tools import Q
# from Copulas.fgm import FGM
# 
# q = Q(u, v, FGM, [alpha])
# 
# qnC = Qn(
#         pd.DataFrame([u,v],index=["x","y"]).T,
#         u,
#         v,
#         Cn
#         )
# 
# qnD = Qn(
#         pd.DataFrame([u,v],index=["x","y"]).T,
#         u,
#         v,
#         Dn
#         )
# 
# print((abs(q)>1).sum(), (abs(qnC)>1).sum(),
#       (abs(qnD)>1).sum())
# 
# TC = ((q-qnC)**2).sum()
# TD = ((q-qnD)**2).sum()
# 
# print("\n" ,TC, "\n", TD)
# 
# #%%
# 
# q = Q(u, v, Gumbel_Barnett, [alpha])
# 
# qnC = Qn(
#         pd.DataFrame([u,v],index=["x","y"]).T,
#         u,
#         v,
#         Cn
#         )
# 
# qnD = Qn(
#         pd.DataFrame([u,v],index=["x","y"]).T,
#         u,
#         v,
#         Dn
#         )
# 
# TC = ((q-qnC)**2).sum()
# TD = ((q-qnD)**2).sum()
# 
# print((abs(q)>1).sum(), (abs(qnC)>1).sum(),
#       (abs(qnD)>1).sum())
# 
# print("\n" ,TC, "\n", TD)
# 
# =============================================================================
