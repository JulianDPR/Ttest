# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:58:24 2023

@author: Julian
"""

#%% Import packages

import numpy as np
import pandas as pd
import statsmodels.stats as sta
import statsmodels.formula.api as sfa
import statsmodels.api as sa
import scipy.stats as ss
import scipy.optimize as so
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd
import threading as th
from distfit import distfit

#%% Copulas modules
from Copulas.clayton import Clayton
from Copulas.gumbel_hougaard import Gumbel_Hougaard
from Copulas.fgm import FGM
from Copulas.gumbel_barnett import Gumbel_Barnett
from Copulas.frank import Frank

#%% Otras

def Escritor(Doc,DataFrame,Name,index=True,over_repla="overlay",srow=0,scol=0,mode="a"):
  #Doc: Ruta
  #DataFrame: Datos
  #Name: Nombre del sheet
  #index: Incluir index ó no, por defecto es True
  #over_repla: Sobreescribir ó reemplazar
  if mode == "a":
    writer_ = pd.ExcelWriter(Doc, mode=mode,if_sheet_exists=over_repla)
  else:
    writer_ = pd.ExcelWriter(Doc, mode=mode)
  DataFrame.to_excel(writer_, sheet_name=Name,index=index, startrow = srow, startcol = scol)
  writer_.close()

#%% Ranks

def ranks(XY):
    
    """
    XY : DataFrame with n samples u and v
    
    Return : The ranks statistics
    of u and v and lenght sample
    """
    try: 
         
        n = XY.shape[0]
        
        #R = XY.sort_values(by="x").reset_index().reset_index().sort_values(by="index").level_0.values + 1
        
        R = XY.sort_values(by='x').reset_index().sort_values(by='index').reset_index().groupby('x')[['level_0']].transform('mean').values.ravel()+1
        
        #S = XY.sort_values(by="y").reset_index().reset_index().sort_values(by="index").level_0.values + 1
        
        S = XY.sort_values(by='y').reset_index().sort_values(by='index').reset_index().groupby('y')[['level_0']].transform('mean').values.ravel()+1
        
        return n, R, S
    
    except:
        
        raise Exception("DF with columns names (x,y)")

#%% Escale the sample
        
def u_v(XY):
    
    """
    XY : DataFrame with n samples x and y
    
    Return : u and v
    """
    
    n, R, S = ranks(XY)
    
    return R/(n+1), S/(n+1)    
    

#%% Empirical copula

# Biased

def Dn(XY, u, v):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    return: Classic empirical cumulative distribution function
    
    """
    
    n, R, S = ranks(XY)
    
    dn = np.sum((R / (n) <= u[:, None]) & (S / (n) <= v[:, None]),
                axis=1)/n
    
    return dn
    
# Unbiased

def Cn(XY, u, v):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    return: Modificatedt empirical cumulative distribution function
    
    """
    
    n, R, S = ranks(XY)
    
    cn = np.sum((R / (n + 1) <= u[:, None]) & (S / (n + 1) <= v[:, None]),
                axis=1)/n
    
    return cn

#%% Empirical q

def Qn(XY, u, v, function):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    functon: The empirical Copula that you want to use Cn or Dn
    
    return: Empirical quadrate concordance coefficient
    
    """
    
    w = np.sqrt(u*(1-u)*v*(1-v))
    
    pi = u*v
    
    fn = function(XY, u, v)
    
    return (fn-pi)/w

#%% Real q

def Q(u, v, Cfunction, parameters):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    cfuncton: The real copula that you want to evalue
    
    parameters: A vectro with estimate parameters
    
    return: Empirical quadrate concordance coefficient
    
    """
    
    w = np.sqrt(u*(1-u)*v*(1-v))
    
    pi = u*v
    
    f = Cfunction(pd.DataFrame([u,v],index=['x','y']).T, *parameters)
    
    return (f-pi)/w


#%% Sample generator

def sample(n, key, alpha, threading = True):
    
    """
    n: u, v sample size
    
    key: Copula generate name
    
    alpha: Copula real parameter
    
    return: u, v sample
    
    """
        
    v1 = ss.uniform(0,1).rvs(n)
    
    v2 = ss.uniform(0,1).rvs(n)
    
    if key == 'clayton':
        
        from Copulas.clayton import inv_Clayton
        
        u = v1
        
        v = inv_Clayton(u, v2, alpha)
        
    elif key == "gumbel_hougaard":
        
        from Copulas.gumbel_hougaard import inv_Gumbel_Hougaard
        
        x = ss.levy_stable(1/alpha,
                            1,
                            alpha==1,# Si ocurre entonces 1, caso contrario 0
                    np.cos(np.pi/(2*alpha))**alpha).rvs(n)
        
        u, v = inv_Gumbel_Hougaard(v1, v2, x, alpha)
        
    elif key == "frank":
        
        from Copulas.frank import inv_Frank
        
        u = v1
        
        v = inv_Frank(u, v2, alpha)
        
    elif key == "gumbel_barnett":
        
        from Copulas.gumbel_barnett import inv_Gumbel_Barnett
        
        n__ = np.array([
             36,
             56,
             104,
             504,
             1000
             ])
        
        n__= n__[n__>=n][0]    
        
        v1 = ss.uniform(0,1).rvs(n__)
        
        v2 = ss.uniform(0,1).rvs(n__)
        
        Th = []
        
        result = []
        
        k = 8
        
        n_ = (n__)//k
        
        if threading:
        
            def quick(g, x):
            
                x.append(inv_Gumbel_Barnett(v1[g[0]:g[1]],
                                        v2[g[0]:g[1]],
                                        alpha))

            for i in range(k):
            
                thread = th.Thread(target = quick,
                               args = ([n_*i, n_*(i+1)],
                                       result))
                
                Th.append(thread)
        
                thread.start()
            
            for i in Th:
            
                i.join()
        
           # print(result)
            
            result = np.array(result).ravel()
        
            u = v1[:n]
        
            v = result[:n]
# =============================================================================
#         
#             u = u[abs(v)<=1][:n]
#             
#             v = v[abs(v)<=1][:n]
# =============================================================================
        
            #print(u, v)
        
        else:
            
            u = v1            
                   
            v = inv_Gumbel_Barnett(v1, v2, alpha)
            
            
            
            #print(u, v)
        
    elif key == "fgm":
        
        from Copulas.fgm import inv_FGM
        
        u, v = inv_FGM(v1, v2, alpha)
        
    else:
        
        raise Exception("La copula no esta dentro del estudio")
        
    #print(len(u), len(v), n__)    
    
    return u,v
#%% max log-likehood

def c(u, v, function ,alpha):
    
    """
    u, v: random uniform variables (0,1)
    
    function: Real copula function
    
    alpha: Copula real parameter
    
    return: Density copula function
    
    """
    
    f = np.vectorize(lambda u,v: nd.Hessian(lambda XY: function(pd.DataFrame([XY[0],XY[1]], index=['x','y']).T, alpha))([u,v]).diagonal(1))(u,v).ravel()

    return f[~np.isnan(f)]

def mom(u, v, function, alpha):
    
    """
    u, v: random uniform variables (0,1)
    
    function: Real copula function
    
    alpha: Copula real parameter
    
    return: The moment estimated
    
    """
    
    return c(u, v, function, alpha).mean()
    
def clog_likelihood(alpha, function, XY):
    
    """
    XY: random vector uniform variables from u, v
    
    function: Real copula function
    
    return: The log-likelihood function
    
    """
    
    return -np.sum(np.log(c(XY.x, XY.y, function, alpha)))

def log_likelihood(theta, function, sample):
    
    """
    
    sample: m samples from the T distribution
    
    function: possible density function for T
    
    theta: parameters vector
    
    """
    
    if function == ss.genextreme:
    
        return -np.sum(np.log(function(0, theta[0], theta[1]).pdf(sample)))
    
    elif function == ss.gamma:
        
        return -np.sum(np.log(function(theta[0], 0, theta[1]).pdf(sample)))
    
    elif function == ss.lognorm:
        
        return -np.sum(np.log(function(0, theta[0], theta[1]).pdf(sample)))

def max_ll(XY, function, alpha, fun = clog_likelihood):
    
    """
    XY: random vector uniform variables from u, v
    
    function: Real copula function
    
    alpha: Seed of posible copula real parameter
    
    fun: The log-likelihood function
    
    return: Return the max(alpha) for the select copula
    
    """
    
    return np.array(so.fmin(fun, alpha, args=(function, XY)))

#u, v = sample(1000, 'clayton', 3)

#max_ll(pd.DataFrame([u, v], index=['x','y']).T, Clayton, 2)


#%% T generator

def T_gen(n, key, alpha, empcop = Cn, threading = True, dist_key = None, dist_parameter = None, graph = False):
    
    """
    n: u, v sample size
    
    Key: Copula name
    
    alpha: Copula real parameter
    
    empcop: The empirical Copula that you want to use Cn or Dn
    
    return: A sample from T distribution
    
    """
    
    if dist_key == None:
        
        dist_key = key
        
        dist_parameter = alpha
    
    u, v = sample(n, key, alpha, threading)
    
    XY = pd.DataFrame(dict(x = u, y = v))

    qn = Qn(XY, u, v, empcop)
    
    if dist_key == "clayton":
        
        from Copulas.clayton import Clayton
        
        function = Clayton
    
    elif dist_key == "gumbel_hougaard":
        
        from Copulas.gumbel_hougaard import Gumbel_Hougaard
    
        function = Gumbel_Hougaard
    
    elif dist_key == "gumbel_barnett":
        
        from Copulas.gumbel_barnett import Gumbel_Barnett
        
        function = Gumbel_Barnett
        
    elif dist_key == "fgm":
        
        from Copulas.fgm import FGM
        
        function = FGM
        
    elif dist_key == "frank":
        
        from Copulas.frank import Frank
        
        function = Frank
        
    else:
        
        raise Exception("La función no está dentro del estudio")

    q = Q(u, v, function, [dist_parameter])
    
    qn_ = qn[(abs(q)<=1)&(abs(qn)<=1)]
    
    q_ = q[(abs(q)<=1)&(abs(qn)<=1)]
    
    
    if graph:
        
        fitdist((q_-qn_)**2, "T sample")
    
   # fitdist(np.sqrt(len(q_))*(qn_-q_))
    
    return ((q_-qn_)**2).sum()
#%%

def t_gen(m, n, key, alpha, return_ ,empcop = Cn, threading = True):
    
    """
    m: T sample size
    
    n: u, v sample size
    
    Key: Copula name
    
    alpha: Copula real parameter
    
    return_: Save array
    
    empcop: The empirical Copula that you want to use Cn or Dn
    
    return: m samples from T distribution
    
    """
    
    np.random.seed(1914)
    
    return_.append(list(map(lambda x: T_gen(n, key, alpha, Cn, threading), range(m))))


#%% Distfit

def fitdist(sample, title = None):
    
    """
    sample: m samples form T distristribution
    
    Title: Name for plot or file with copula names
    
    return: A set with posibles distributions for T
    
    """
    
    fig, ax = plt.subplots(figsize=(20,15))
    
    dfit = distfit()
    
    results = dfit.fit_transform(sample)
    
    dfit.plot(n_top = 3, ax = ax)
    
    if title != None:
        
        ax.set_title(title, fontsize = 16)
    
    return fig, ax, results


    
#%% Bootstrapping 
def Bootst(sample, k):
    
    np.random.seed(1914)
    
    """
    sample: m sample from T distribution
    
    k: number of m size resamples 
    
    return: A df with bias, var and mse
    """
    
    sk = ss.skew(sample)
    
    if sk != 0:
        
        stat = np.median

    else:
        
        stat = np.mean

    m = len(sample)
    
    botst = np.zeros((m, k))
    
    bootst = np.apply_along_axis(
        lambda x: np.random.choice(sample,
                                   m,
        replace = True) if np.sum(x)== 0 else x,
        0, 
        botst  
        )
    
    T = pd.DataFrame(bootst).apply(moms)
    
    ET = T.mean(axis = 1)
    
    VT = T.var(axis = 1)
    
    
    VT = pd.Series(dict(Vgamma = VT.iloc[0]+VT.iloc[1],
              Vln = VT.iloc[2]+VT.iloc[3],
              Vgum = VT.iloc[4]+VT.iloc[5]))

    b = pd.concat([ET,VT])
    
    return b

def Bootst2(sample, k):
    
    """
    sample: m sample from T distribution
    
    k: number of m size resamples 
    
    return: A array of k size from the T distribution average resamples
    """
    
    sk = ss.skew(sample)
    
    if sk != 0:
        
        stat = np.median

    else:
        
        stat = np.mean

    m = len(sample)
    
    botst = np.zeros((m, k))
    
    bootst = np.apply_along_axis(
        lambda x: np.random.choice(sample,
                                   m,
        replace = True) if np.sum(x)== 0 else x,
        0, 
        botst  
        )
    
    b = np.quantile(bootst, np.array([0.025, 0.975]), axis = 0)
    
    return stat(b, axis = 1)

def Bootst3(sample, k):
    
    """
    sample: m sample from T distribution
    
    k: number of m size resamples 
    
    return: A array of k size from the T distribution average resamples
    """
    
    sk = ss.skew(sample)
    
    if sk != 0:
        
        stat = np.median

    else:
        
        stat = np.mean

    m = len(sample)
    
    botst = np.zeros((m, k))
    
    bootst = np.apply_along_axis(
        lambda x: np.random.choice(sample,
                                   m,
        replace = True) if np.sum(x)== 0 else x,
        0, 
        botst  
        )
    
    b = np.mean(bootst, axis = 0)
    
    return b
    

def moms(sample):
    
    """
    sample: A array with n repeats from x's distribution
    
    return: Gamma, Gumbel, Lognormal parameters estimators
    
    """
    
    dic = dict(
    
    gamma_a = np.mean(sample)**2/(np.var(sample, ddof=1)),
    
    gamma_b = np.mean(sample)/(np.var(sample, ddof=1)),
    
    ln_m = np.log(np.mean(sample)**2/(np.sqrt(np.var(sample, ddof = 1)+
                                            np.mean(sample)**2))),
    ln_b = np.log(
        (np.var(sample, ddof = 1)+np.mean(sample)**2)/np.log(
            np.mean(sample))
        ),
    
    gum_m = np.mean(sample)-np.sqrt(6)*np.euler_gamma*np.std(sample,
                                                ddof=1)/np.pi,
    gum_b = np.std(sample, ddof = 1)*np.sqrt(6)/np.pi
    
    )
    
    return pd.Series(dic)

def gum_mom(sample):
    
    """
    sample: A array with n repeats from x's distribution
    
    return: The Gumbel parameters estimators
    
    """
    
    dic = dict(
        gum_m = np.mean(sample)-np.sqrt(6)*np.euler_gamma*np.std(sample,
                                                    ddof=1)/np.pi,
        gum_b = np.std(sample, ddof = 1)*np.sqrt(6)/np.pi
        
        )
    
    return pd.Series(dic)

def bias(sample):
    
    """
    sample: A array with n repeats from x's distribution
    
    return: The Variance, Bias and Mse from X estimators
    """
    
    moms_ = sample.apply(moms)
    
    bootst = sample.apply(lambda x: Bootst(x, 1000))
    
    r_bootst = bootst.loc[moms_.index,:]
    
    bias = (moms_-r_bootst)
    
    bias = pd.DataFrame(dict(
        Bgamma = bias.iloc[[0,1],:].apply(lambda x: np.linalg.norm(x, axis = 0)**2),
        Bln = bias.iloc[[2,3],:].apply(lambda x: np.linalg.norm(x, axis = 0)**2),
        Bgum = bias.iloc[[4,5],:].apply(lambda x: np.linalg.norm(x, axis = 0)**2)
        )).T

    var = bootst.loc[bootst.index[6:],:]
    
    var_ = var.copy()
    
    var_.index = bias.index
    
    mse = bias+var_
    
    mse.index = ["Mgamma", "Mln", "Mgum"]
    
    return dict(bias = bias, var = var, mse = mse)

def ajuste(sample, parameters):
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    qq = ss.probplot(sample, sparams=(0, *parameters),
                dist = ss.genextreme, plot = ax)
    
    np.random.seed(1914)
    
    ad = ss.anderson_ksamp([qq[0][1], qq[0][0]
                            ]).pvalue
    
    ax.set_title("AD-Pvalue: %.4f"%(ad))
    
    fig.suptitle(f"{sample.name}")
    
    ax.set_xlabel("Cuantil teórico")
    
    ax.set_ylabel("Cuantil ordenado")
    
    fig.tight_layout()
    
    fig.show()
    
    return ad
    
# =============================================================================
# def root(function, args):
#     
#     """
#     function: La función a la cual se le buscan las raíces
#     args: Argumentos extras de la función
#     regresa: El x donde la función es 0
#     """
#     
#     
# =============================================================================

#%% Jack knife



# =============================================================================
# #%% Try
# 
# xy = {"x":ss.norm(0,1).rvs(100), "y":ss.norm(0,2).rvs(100)}
# 
# df = pd.DataFrame(xy)
# 
# print(Dn(df, 0.8, 0.8))
# print(Cn(df, 0.8, 0.8))
# print(Qn(df, 0.999, 0.999, Cn))
# 
# #%%
# 
# xs, ys = np.meshgrid(np.linspace(0.001,0.999,100), np.linspace(0.001,0.999,100))
# 
# qn = Qn(df, xs, ys, Dn)
# 
# DF = pd.DataFrame([xs.ravel(), ys.ravel(),
#                    qn.ravel()], index = ['X', 'Y', 'Qn']).T
# 
# DF.Qn = DF.Qn.apply(lambda x: np.nan if abs(x)>1 else x)
# 
# #%%
# 
# sns.heatmap(DF.Qn.to_numpy().reshape(*qn.shape), vmin=-1, vmax=1)
# 
# 
# =============================================================================

