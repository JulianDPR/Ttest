# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:18:59 2024

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
from tools import ajuste
#%% Objetivo 1
path = "C:/Users/Bienvenido/Desktop/TG/Imag"

Tg = True

n = [
     30,
     50,
     100,
     500,
     1000
     ]

m = 1100#//k

names = [
   "gumbel_barnett",
    "clayton",
    "frank", 
    "gumbel_hougaard",
    "fgm"
         ]

dependence =  [
            "weak",
           "moderate",
           "strong"
           ]

parameters = pd.DataFrame(
    [
     [.2, .5, .9],
     [0.5, 2, 8],
     [1.86, 5.73, 18.19],
     [1.25, 2, 5],
     [.2, .5, .9]
     ],
    index = names,
    columns = dependence
    ).T

parameters = parameters.to_dict()

#%%

n_, name_, pars_ = np.meshgrid(n, names, dependence)

n_, name_, pars_ = n_.ravel(), name_.ravel(), pars_.ravel()

if __name__ == "__main__":
    
    T_samples = {}
    
    T_sample = {}
    
    start = t.time()
    
    for i,j,k in zip(n_, name_, pars_):
        
        if i == 30 and k == "weak":
            
            T_sample.update({j:{}})
        
        #k = th.active_count()
        
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

        t_gen(m, i, j, parameters[j][k], results, threading=Tg)
        
        T = (np.array(results).ravel())[:1000]
        
        T_samples.update({f"{i}_{k}_{j}":T})
        
        T_sample[j].update({k:{}})
        
        T_sample[j][k].update({f"{i}":T})
        
        print(T_sample[j][k])
        
        fig, ax, results = fitdist(T, f"{i}_{k}_{j}")
        
        fig.savefig(path+"/"+j+"/"+f"{i}_{k}"+".png", dpi = 300)
        
        fig.show()
        
        if f"{i}_{k}" == "50_weak":
            
            Escritor(path+"/"+j+"/"+"resultados.xlsx", results["summary"], Name=f"{i}_{k}", index = False, 
                     mode = "w")
            
            Escritor(path+"/"+j+"/"+"muestra.xlsx", pd.Series(T), Name=f"{i}_{k}", index = False, 
                     mode = "w")
        
        else:
            
            Escritor(path+"/"+j+"/"+"resultados.xlsx", results["summary"], Name=f"{i}_{k}", index = False)
            
            Escritor(path+"/"+j+"/"+"muestra.xlsx", pd.Series(T), Name=f"{i}_{k}", index = False)

#%% Objetivo 2
    df = pd.DataFrame(T_samples)

    Bias = bias(df)
    
    k = df.apply(gum_mom).T
    
    aj = df.apply(lambda x: ajuste(x, k.loc[x.name,:]))
    
#%% Objetica 3
    
    
    for i,j in zip(k.index, name_):
        
        fig, ax = plt.subplots(figsize = (10, 8))
        
        g = np.exp((k.loc[i,"gum_m"]-df[i])/(k.loc[i,"gum_b"]))
    
        ax.hist((2)*g, bins = 10)
                                                                
        q = np.quantile((2)*g
         , np.array([0.025, 0.975])) 
                                                                
        print(q)                                                           
                                                                
        ax.vlines(q[0], *ax.get_ylim(), color = "r", linestyles = "--")

        ax.vlines(q[1], *ax.get_ylim(), color = "r", linestyle = "--") 

        ax.set_title(i+" "+r"$2 e^{\frac{\mu-T}{\beta}}$") 
        
        fig.savefig(path+"/"+j+"/"+f"{i}"+".png", dpi = 500)

        fig.show()  
        
    end = t.time()

    print("Total Horas de trabajo: %.4f"%((end-start)/60/60))  
    
#%% ¿Cual es el valor de los parametros de T? muy posiblemente sea 0.95, 0.34
# Parece no existir a simple vista sino que dependende de la distribución.


    #n_30 = list(map(lambda x: "30_" in x,(k.index)))
    #n_50 = list(map(lambda x: "50_" in x,(k.index)))
    #n_100 = list(map(lambda x: "100_" in x,(k.index)))
    #n_500 = list(map(lambda x: "500_" in x,(k.index)))
    #n_1000 = list(map(lambda x: "1000_" in x,(k.index)))

   # k_30 = k.loc[n_30, :]
   # k_50 = k.loc[n_50, :]
   # k_100 = k.loc[n_100, :]
   # k_500 = k.loc[n_500, :]
   # k_1000 = k.loc[n_1000, :]

#%%
    #fig, ax = plt.subplots(figsize=(6,4))
    #k_30.plot(kind="line", marker = "o", ax=ax, rot=90)

    #fig.show()

   # fig, ax = plt.subplots(figsize=(6,4))
   # k_50.plot(kind="line", marker = "o", ax=ax, rot=90)

   # fig.show()

    #fig, ax = plt.subplots(figsize=(6,4))
    #k_100.plot(kind="line", marker = "o", ax=ax, rot=90)

    #fig.show()

   # fig, ax = plt.subplots(figsize=(6,4))
   # k_500.plot(kind="line", marker = "o", ax=ax, rot=90)

   # fig.show()

   # fig, ax = plt.subplots(figsize=(6,4))
   # k_1000.plot(kind="line", marker = "o", ax=ax, rot=90)

   # fig.show()

