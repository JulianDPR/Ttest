# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:56:47 2023

@author: Julian
"""

#%% Import oficials packages
#Statics
import scipy.stats as ss
import statsmodels.api as sa
import statsmodels.stats as sts
#Linear algebra and dataframes
import numpy as np
import numpy.linalg as nl
import pandas as pd
#Graphics
import matplotlib.pyplot as plt
import seaborn as sns
#Fit
from distfit import distfit
#
import time as t
import threading as th

#%% Own modules
#from tools
from tools import Cn
from tools import Dn
from tools import Qn
from tools import Q
from tools import ranks
from tools import T_gen
from tools import sample
from tools import fitdist
from tools import Escritor
from tools import t_gen
from tools import moms
from tools import Bootst
from tools import bias
from tools import gum_mom

#%% Objetivo 1

n = 1000

Tg = True

path = "C:/Users/Bienvenido/Desktop/TG/Imag"

#%% Parametros de estudio

parameters = pd.DataFrame(
    [
     [.2, .5, .9],
     [0.5, 2, 8],
     [1.86, 5.73, 18.19],
     [1.25, 2, 5],
     [.2, .5, .9]
     ],
    index = [
        "gumbel_barnett",
        "clayton",
        "frank", 
        "gumbel_hougaard",
        "fgm"
             ],
    columns = [
                "weak",
               "moderate",
               "strong"
               ]
    ).T

parameters = parameters.to_dict()
#%%

if __name__ == "__main__":
    
    T_samples = {}
    
    start = t.time()

    for i in parameters:
        
        for j in parameters[i]:
            
            np.random.seed(1914)
            
            #k = th.active_count()
            
            m = 1100#//k
            
            # Create a results list to store the results
            results = []

            # Create and start a thread for each data chunk
            #threads = []
            
            #for p in range(k):
                
              #  thread = th.Thread(target=t_gen, args=(m, n, i, parameters[i][j], results, Cn))
                
              #  threads.append(thread)
                
              #  thread.start()
                
           # for thread in threads:
                
                #thread.join()
                
            t_gen(m, n, i, parameters[i][j], results, threading=Tg)
            
            T = (np.array(results).ravel())[:1000]
            
            T_samples.update({i+"_"+j:T})
            
            fig, ax, results = fitdist(T, i+j)
            
            fig.savefig(path+"/"+i+"/"+j+".png", dpi = 500)
            
            fig.show()
            
            if j == "weak":
                
                Escritor(path+"/"+i+"/"+"resultados.xlsx", results["summary"], Name=j, index = False, 
                         mode = "w")
                
                Escritor(path+"/"+i+"/"+"muestra.xlsx", pd.Series(T), Name=j, index = False, 
                         mode = "w")
            
            else:
                
                Escritor(path+"/"+i+"/"+"resultados.xlsx", results["summary"], Name=j, index = False)
                
                Escritor(path+"/"+i+"/"+"muestra.xlsx", pd.Series(T), Name=j, index = False)
                
    end = t.time()

    print("Total Horas de trabajo: %.4f"%((end-start)/60/60))  

#%% Objetivo 2
    df = pd.DataFrame(T_samples)

    Bias = bias(df)
    
    #%%
    
    k = df.apply(gum_mom).T
    
    fig, ax = plt.subplots(figsize = (10, 8))
    
    ax.hist(2*k.loc["gumbel_barnett_strong",
                     "gum_b"]*np.exp(
                         (k.loc["gumbel_barnett_strong",
                                "gum_m"]-df.gumbel_barnett_weak)/(k.loc["gumbel_barnett_strong",
                                                 "gum_b"]*k.loc["gumbel_barnett_weak",
                                                        "gum_m"])), bins = 10)
                                                                
    q = np.quantile(
    2*k.loc["gumbel_barnett_strong",
                     "gum_b"]*np.exp(
                         (k.loc["gumbel_barnett_strong",
                                "gum_m"]-df.gumbel_barnett_weak)/(k.loc["gumbel_barnett_strong",
                                                 "gum_b"]*k.loc["gumbel_barnett_strong",
                                                        "gum_m"]))
    , np.array([0.025, 0.975])) 
                                                                
    print(q)                                                           
                                                                
    ax.vlines(q[0], *ax.get_ylim(), color = "r", linestyles = "--")

    ax.vlines(q[1], *ax.get_ylim(), color = "r", linestyle = "--") 

    ax.set_title(r"gumbel_barnett_strong $2\beta e^{\frac{\mu-T}{\mu \beta}}$") 

    fig.show()                                                      

#%%

#import os

#os.system("shutdown -h")






