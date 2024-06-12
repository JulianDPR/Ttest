# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 23:59:40 2024

@author: Julian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

#%% FGM copula

def FGM(XY, alpha, diff = False):
    
    if diff:
        
        x = XY.x.values.ravel()
        y = XY.y.values.ravel().reshape(-1,1)
        
        return x*y*(1+alpha*(1-x)*(1-y))
        
    else:
        
        return XY.x*XY.y*(1+alpha*(1-XY.x)*(1-XY.y))

def inv_FGM(v1, v2, alpha):
    
    A = alpha*(2*v1-1) - 1
    
    B = (1 - 2*alpha*(2*v1 - 1) + alpha**2*(
        2*v1 - 1)**2 + 4*alpha*v2*(2*v1-1))**(1/2)
    
    return v1, 2*v2/(B-A)


#%%
# =============================================================================
# 
# v1 = ss.uniform(0,1).rvs(1000)
# v2 = ss.uniform(0,1).rvs(1000)
# alpha = 0.5
# 
# u, v = inv_FGM(v1, v2, alpha)
# 
# plt.scatter(u, v, alpha=0.5, s=8)
# 
# =============================================================================

# =============================================================================
# #%%
# 
# from tools import Qn
# from tools import Cn
# from tools import Dn
# from tools import Q
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
# TC = ((q-qnC)**2).sum()
# TD = ((q-qnD)**2).sum()
# 
# print("\n" ,TC, "\n", TD)
# 
# #%%
# 
# from Copulas.clayton import Clayton
# 
# q = Q(u, v, Clayton, [alpha])
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
# print("\n" ,TC, "\n", TD)
# 
# 
# 
# 
# 
# =============================================================================
