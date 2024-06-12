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
# Writer
from tools import Escritor
# Sample
from tools import sample
from tools import t_gen
# Statistics
from tools import Cn
from tools import Dn
from tools import Qn
from tools import Q
from tools import ranks
from tools import T_gen
from tools import k_t
# fit distribution
from tools import fitdist
from tools import fit
# Estimation methods
from tools import moms
#from tools import mom2
from tools import Bootst
#from tools import Bootst_2
from tools import bias
#from tools import bias2
from tools import gum_mom
from tools import bias_plot
# Calculus tools 
from findiff import FinDiff
from tools import c3
# Ttest
from tools import Ttest
from tools import Pow_Conf
from tools import T
from tools import fitdist2
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
#%% Showing

for i in parameters.keys():
    
    fig, ax = plt.subplots(1,3,figsize=(7,3))
    
    ax = ax.ravel()
    
    for j,k in zip(parameters[i].keys(),ax):
        u,v = sample(1000, i, parameters[i][j])
        k.scatter(x=u, y=v,
                        marker=".",
                        facecolors="none",
                        edgecolors="darkcyan",
                        #label=r"$\theta = $"+f"{parameters[i][j]}",
                        s=30
                        )
        
        #k.legend(loc="lower right",fontsize=15, facecolor="none")
        k.set_xlabel(r"$\theta = $"+f"{parameters[i][j]}")
    fig.tight_layout()
    
    fig.savefig(path+"/"+i+"/"+f"{i}"+".png", dpi = 200)
    
    fig.show()
#%%

n_, name_, pars_ = np.meshgrid(n, names, dependence)

n_, name_, pars_ = n_.ravel(), name_.ravel(), pars_.ravel()

#%%

if __name__ == "__main__":
    
    T_samples = {}
    
    T_sample = {}

#%%
    
    start = t.time()
    
    for i,j,k in zip(n_, name_, pars_):
        
        c = 0
        
        if j not in T_sample.keys():
            
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

        t_gen(m, i, j, parameters[j][k], results, threading=Tg, random_seed=1927+c)
        
        T = (np.array(results).ravel())[:1000]
        
        T_samples.update({f"{i}_{k}_{j}":T})
        
        if k not in T_sample[j].keys():
        
            T_sample[j].update({k:{}})
        
        T_sample[j][k].update({f"{i}":T})
        
        #print(T_sample[j][k])
        
        fig, ax, results = fitdist(T, f"{i}_{k}_{j}")
        
        plt.close(fig)        
       # fig.savefig(path+"/"+j+"/"+f"{i}_{k}"+".png", dpi = 300)
        
       # fig.show()
        
        if f"{i}_{k}" == f"{min(n)}_weak":
            
            Escritor(path+"/"+j+"/"+"resultados.xlsx", results["summary"], Name=f"{i}_{k}", index = False, 
                     mode = "w")
            
            Escritor(path+"/"+j+"/"+"muestra.xlsx", pd.Series(T), Name=f"{i}_{k}", index = False, 
                     mode = "w")
        
        else:
            
            Escritor(path+"/"+j+"/"+"resultados.xlsx", results["summary"], Name=f"{i}_{k}", index = False)
            
            Escritor(path+"/"+j+"/"+"muestra.xlsx", pd.Series(T), Name=f"{i}_{k}", index = False)
        
        c += 1

#%% Objetivo 1.1

    espaniol = {"weak":"Débil", "moderate":"Moderada", "strong":"Fuerte"}

    dependencia, nombres, muestra = np.meshgrid(dependence, names, n)

    dependencia, nombres, muestra = dependencia.ravel(), nombres.ravel(), muestra.ravel()

    for i,j,k in zip(dependencia, nombres, muestra[::-1]):
    
        if k == np.max(muestra):
        
            fig, ax = plt.subplot_mosaic([["left","left","right1","right2"],
                                      ["left", "left", "right3", "right4"]]
                                     , figsize=(12, 6))
        
            fitdist2(T_sample[j][i][f"{k}"], f"n={k} (Dependencia: {espaniol[i]})" ,ax["left"])
        
            ax["left"].set_xlabel("Valores")
            ax["left"].set_ylabel("Frecuencias")
        
        elif k > np.min(muestra):
        
            fitdist2(T_sample[j][i][f"{k}"],f"n={k}" ,
                     ax[list(ax.keys())[list(np.unique(muestra))[::-1].index(k)]])
          
            ax[list(ax.keys())[list(np.unique(muestra))[::-1].index(k)]].set_ylabel("")
            ax[list(ax.keys())[list(np.unique(muestra))[::-1].index(k)]].set_xlabel("")
            ax[list(ax.keys())[list(np.unique(muestra))[::-1].index(k)]].legend(prop={'size': 4.5})
            
        
        else:
        
            fitdist2(T_sample[j][i][f"{k}"],f"n={k}" ,
                 ax[list(ax.keys())[-1]])
          
            ax[list(ax.keys())[-1]].set_ylabel("")
            ax[list(ax.keys())[-1]].set_xlabel("")
            
            ax[list(ax.keys())[-1]].legend(prop={'size': 4.5})
            
            fig.tight_layout()
        
            fig.savefig(path+"/"+j+"/"+f"Resumen_ajuste_{j}{i}"+".png", dpi = 150)
        
            fig.show()
        
        
#%% Objetivo 2

# Cuales son las distribuciones que mejor se ajustan
# El sesgo de cada uno de los parametros de la distribucion
# La eficiencia y consistencia del parametro

    df = pd.DataFrame(T_samples)
    
    Bias = bias(df)
    #%%
    print((Bias["mse"].T[["Mln","Mgum"]].groupby([name_,pars_]).apply(lambda x: ((x==x.min(axis=1).values.reshape(-1,1))).mean())).to_latex(multirow=True,
                                                                                 caption="Resumen calidad de estimación",
                                                                                 label="tab:calidad"))
    print(Bias["mse"].T[["Mln","Mgum"]].groupby([name_,pars_]).apply(lambda x: ((x==x.min(axis=1).values.reshape(-1,1))).mean()).mean())
    #%%
    k = df.apply(moms).T
    
    print(k.groupby([name_,pars_]).mean().round(2).to_latex(multirow=True,
                                                            caption="Valor medio de la estimación de parámetros",
                                                            label="tab:estimacion"))
    
    print(k.groupby([name_,pars_]).median().round(2))
    
    k.groupby([name_,pars_]).median().plot(kind="line",marker=".", rot=90)
    
    plt.show()
    
    aj = df.apply(lambda x: fit(x, k.loc[x.name,:]))

#%%
    print((aj.T.groupby([name_,pars_]).apply(lambda x: np.sum(x>0.05))).to_latex(multirow=True,
                                                                                 caption="Resumen pruebas Anderson-Darling",
                                                                                 label="tab:Anderson"))
    
#%%

    #df.apply(lambda x: fit(x, k.loc[x.name,["ln_m","ln_b"]], True))
    
    #print((Bias["mse"].T.values[:,[0,2]] >= Bias["mse"].T.values[:,1][:,None]).mean(axis=0).mean())
    
    pars__, name__ = np.meshgrid(np.unique(pars_),np.unique(name_))
    
    name__, pars__ = name__.ravel(), pars__.ravel()
    
    result_ = {}
    
    for i,j in zip(name__, pars__):
        
        result_.update({i+"-"+j:bias_plot(pd.DataFrame(pd.DataFrame(T_sample).loc[j,
                                    i]), ""
                                        #  i.upper()+"-"+j.upper()
                                          , path+"/"+i+"/"+f"{i}_{j}"+".png")})
    
#%% Objetica 3

# Intervalos de confianza (Distribución y empíricos)

    IE = df.quantile([0.025,0.975]).T

    Iln = np.array(ss.lognorm(k.iloc[:,3]**(1/2),
                              0,
        np.exp(k.iloc[:,2])).interval(0.95)).T

    Igum = np.array(ss.genextreme(0,
        k.iloc[:,4],
        k.iloc[:,5]).interval(0.95)).T

# Prueba de hipótesis (Distribución y empíricos)

    u,v = sample(100,"clayton", 2)

    r = Ttest(pd.DataFrame(dict(x=u,y=v)),
              "frank",
              RD = True)
    
    print(r.T)
# Confianza según la muestra (Comparativa con Distribución ,Genest y River)
# Potencia según la muestra (Comparativa con Distribución ,Genest y River)
    

#%% Generar escenarios de simulación
   
