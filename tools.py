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
from findiff import FinDiff
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
    
    return R/(n), S/(n)    
    

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
    
    cn = np.sum((R / (n + 1) <= u[:, None]) & ((S / (n + 1)) <= v[:, None]),
                axis=1)/n
    
    return cn

#%% Empirical q

def Qn(XY, u, v, function, GR = False):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    functon: The empirical Copula that you want to use Cn or Dn
    
    return: Empirical quadrate concordance coefficient
    
    """
    
    w = np.sqrt(u*(1-u)*v*(1-v))
    
    pi = u*v
    
    fn = function(XY, u, v)
    
    if GR:
        
        return fn
    
    else:
        
        return (fn-pi)/w

#%% Real q

def Q(u, v, Cfunction, parameters, diff = False ,GR = False):
    
    """
    XY: Dataframe with n samples from x and y
    
    u, v: Random uniform variables in range (0,1)
    
    cfuncton: The real copula that you want to evalue
    
    parameters: A vectro with estimate parameters
    
    return: Empirical quadrate concordance coefficient
    
    """
    
    w = np.sqrt(u*(1-u)*v*(1-v))
    
    pi = u*v
    
    f = Cfunction(pd.DataFrame([u,v],index=['x','y']).T, *parameters, diff)
    
    
    if GR:
        
        return f
    
    else:
        
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
             1000,
             10000
             ])
        
        n__= n__[n__>=n][0]    
        
        v1 = ss.uniform(0,1).rvs(n__)
        
        v2 = ss.uniform(0,1).rvs(n__)
        
        Th = []
        
        result = {}

        
        k = 8
        
        n_ = (n__)//k
        
        if threading:
        
            def quick(g, x, i):
                
                x.update({i:inv_Gumbel_Barnett(v1[g[0]:g[1]],
                                        v2[g[0]:g[1]],
                                        alpha)})

            for i in range(k):
            
                thread = th.Thread(target = quick,
                               args = ([n_*i, n_*(i+1)],
                                       result, i))
                
                Th.append(thread)
        
                thread.start()
            
            for i in Th:
            
                i.join()
        
           # print(result)
           
            result = [result[i] for i in np.sort(list(result.keys()))]
            
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
    
    #import inspect
    
    """
    u, v: random uniform variables (0,1)
    
    function: Real copula function
    
    alpha: Copula real parameter
    
    return: Density copula function
    
    """
    if len(np.array([u]).ravel())==1:
        
        F = lambda x, y, a: function(pd.DataFrame(dict(x=x,y=y), index = [0]), a)
    
    else:
        
        F = lambda x, y, a: function(pd.DataFrame(dict(x=x,y=y)), a)
    
    d_x = nd.Derivative(F, method = "backward")
    
    d_x_y = nd.Derivative(lambda y, x, a: d_x(x, y, a), method = "backward")
    
    f = d_x_y(v, u, alpha)
    
    #f[np.isnan(f)] = 0

    return f#[~np.isnan(f)]

def c2(u ,v, function, alpha):
    
    """
    u, v: random uniform variables (0,1)
    
    function: Real copula function
    
    alpha: Copula real parameter
    
    return: Density copula function
    
    """
    
    du = u[1]-u[0]
    
    #F = lambda x, y, a: function(pd.DataFrame(dict(x=x, y=y)), a, diff = True)
    
    d_xy = FinDiff((0, du), (1, du))
    
    return d_xy(function(pd.DataFrame(dict(x=u, y=v), index = list(range(len(u)))), alpha, diff=True))


def c3(u, v, function, alpha, d = 1e-6, diff = False):
    
    """
    u, v: random uniform variables (0,1)
    
    function: Real copula function
    
    alpha: Copula real parameter
    
    d: Differencial
    
    return: Density copula function
    
    """
    
    #d = 1e-6
    
    f = lambda x, y: function(pd.DataFrame(dict(x=x, y=y), index = list(range(len(np.array(np.ravel([u])))))),
                              alpha, diff) 
    
    d_xy = lambda x, y, d : (((f(x+d,y+d)-f(x-d,y+d))/(2*d))-((f(x+d,y-d)-f(x-d,y-d))/(2*d)))/(2*d)
    
    return d_xy(u,v,d)
#%% Cosas a implementar luego
#El uso de la función arquimediana para simular

#%%
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
#%% Integral
def CopulaIM(function, alpha, nint = 300):
    
    x,y = np.linspace(1e-5,
                       1,
                       nint, endpoint=False),np.linspace(1e-5,
                                         1, nint, endpoint=False)
        
    return (function(pd.DataFrame(dict(x=x, y=y)),
                    alpha, diff = True)*c2(x, y,
                             function,
                             alpha)).mean()
                                           
def CopulaIM2(function, alpha, nint = 300):
    
    x,y = np.linspace(1e-5,
                       1,
                       nint, endpoint=False),np.linspace(1e-5,
                                         1, nint, endpoint=False)
                                         
    return (function(pd.DataFrame(dict(x=x, y=y)),
                    alpha, diff = True)*c3(x, y,
                             function,
                             alpha,diff=True)).mean()                                   
    
#%% T generator

def T_gen(n, key, alpha, empcop = Cn, threading = True, dist_key = None, dist_parameter = None, graph = False, GR = False):
    
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
    
    u_ = u[(abs(q)<=1)&(abs(qn)<=1)]
    
    v_ = v[(abs(q)<=1)&(abs(qn)<=1)]
    
    pi = u_*v_
    
    pi_ = (1-u_)*(1-v_)
    
    if graph:
        
        fitdist((q_-qn_)**2, "T sample")
    
   # fitdist(np.sqrt(len(q_))*(qn_-q_))
   
    if GR:
    
        return (pi*pi_*(q_-qn_)**2).sum()
       
    else:
        
        return ((q_-qn_)**2).sum()


def t_gen(m, n, key, alpha, return_ ,empcop = Cn, threading = True, GR = False, random_seed = 1927):
    
    """
    m: T sample size
    
    n: u, v sample size
    
    Key: Copula name
    
    alpha: Copula real parameter
    
    return_: Save array
    
    empcop: The empirical Copula that you want to use Cn or Dn
    
    return: m samples from T distribution
    
    """
    
    np.random.seed(random_seed)
    
    return_.append(list(map(lambda x: T_gen(n, key, alpha,
                                            Cn, threading,
                                            GR = GR), range(m))))

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

def fitdist2(sample, title="", axis = None):
    
    """
    sample: m samples form T distristribution
    
    Title: Name for plot or file with copula names
    
    return: A set with posibles distributions for T
    
    """
    
    if axis == None:
        
        return distfit(sample, title)
    
    else:
        
        dfit = distfit()
        
        results = dfit.fit_transform(sample)
        
        dfit.plot(n_top = 3, ax = axis)
        
        axis.set_title(title, fontsize = 16)
        
        return results


    
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

# =============================================================================
# def Bootst_2(sample, k):
#     
#     np.random.seed(1914)
#     
#     """
#     sample: m sample from T distribution
#     
#     k: number of m size resamples 
#     
#     return: A df with bias, var and mse
#     """
#     
#     sk = ss.skew(sample)
#     
#     if sk != 0:
#         
#         stat = np.median
# 
#     else:
#         
#         stat = np.mean
# 
#     m = len(sample)
#     
#     botst = np.zeros((m, k))
#     
#     bootst = np.apply_along_axis(
#         lambda x: np.random.choice(sample,
#                                    m,
#         replace = True) if np.sum(x)== 0 else x,
#         0, 
#         botst  
#         )
#     
#     T = pd.DataFrame(bootst).apply(mom2)
#     
#     ET = T.mean(axis = 1)
#     
#     VT = T.var(axis = 1)
#     
#     
#     VT = pd.Series(dict(Vgamma = VT.iloc[:3].sum(),
#               Vln = VT.iloc[3:6].sum(),
#               Vgum = VT.iloc[6:].sum()))
# 
#     b = pd.concat([ET,VT])
#     
#     return b
# =============================================================================

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

def Bootsttrap(sample, k):
    
    m = len(sample)
    
    botst = np.zeros((m, k))
    
    bootst = np.apply_along_axis(
        lambda x: np.random.choice(sample,
                                   m,
        replace = True) if np.sum(x)== 0 else x,
        0, 
        botst  
        )
    
    return bootst
    
#%% Estimation

def moms(sample):
    
    """
    sample: A array with n repeats from x's distribution
    
    return: Gamma, Gumbel, Lognormal parameters estimators
    
    """
    
    dic = dict(
    
    gamma_a = np.mean(sample)**2/(np.var(sample, ddof=1)),
    
    gamma_b = np.mean(sample)/(np.var(sample, ddof=1)),
    
    ln_m = -(1/2)*np.log((np.var(sample,
                                 ddof=1)+
                          np.mean(sample)**2)/(np.mean(sample)**4)),
    ln_b = np.log((np.var(sample,
                          ddof=1)+
                   np.mean(sample)**2)/np.mean(sample)**2),
    
    gum_m = np.mean(sample)-np.sqrt(6)*np.euler_gamma*np.std(sample,
                                                ddof=1)/np.pi,
    gum_b = np.std(sample, ddof = 1)*np.sqrt(6)/np.pi
    
    )
    
    #print(np.mean(sample))
    
    return pd.Series(dic)

# =============================================================================
# def mom2(sample):
#     
#     """
#     sample: A array with n repeats from x's distribution
#     
#     return: Gamma, Gumbel, Lognormal parameters estimators
#     
#     """
#     
#     gam = ss.gamma.fit(sample)
#     
#     gen = ss.genextreme.fit(sample)
#     
#     ln = ss.lognorm.fit(sample)
#     
#     dic = dic = dict(
#     
#     gamma_a = gam[0],
#     
#     gamma_b = gam[1],
#     
#     gamma_c = gam[2],
#     
#     ln_m = ln[0],
#     ln_b = ln[1],
#     ln_c = ln[2]
#     ,
#     gen_a = gen[0],
#     gen_b = gen[1],
#     gen_c = gen[2]
#     
#     )
#     
#     return pd.Series(dic)
# =============================================================================
    

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
    
    #print(moms_)
    
    bootst = sample.apply(lambda x: Bootst(x, 1000))
    
    r_bootst = bootst.loc[moms_.index,:]
    
    bias = (moms_-r_bootst)
    
    #print(bias)
    
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

def bias_plot(sample, title = "", path = None):
    """
    sample: A array with n repeats from x's distribution
    
    return: The plot of Variance, Bias and Mse from X estimators
    """
    
    Bias = bias(sample)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    
    ax = ax.ravel()
    
    var = Bias["var"].T[["Vln","Vgum"]]
    
    mse = Bias["mse"].T[["Mln","Mgum"]]
    
    n = np.array(list(var.index), dtype="float64").ravel()
    
    #print(n)
    
    ((n**(1/2)).reshape(-1,1)*var).plot(kind = "line",
                                        marker = ".",
                                        ax = ax[0], color=["teal", "tomato", "springgreen"])
    
    ax[0].set_title(r"Eficiencia $\lim_{n\to\infty}\sqrt{n}*var[{\hat\Lambda}]$")
    
    ax[0].grid()
    
    mse.plot(kind = "line", marker = ".", ax = ax[1],  color=["teal", "tomato", "springgreen"])
    
    ax[1].set_title(r"Consistencia $E[({\hat\lambda}-\hat\Lambda)^{2}]$")
    
    ax[1].grid()
    
    fig.suptitle(title)
    
    fig.tight_layout()
    
    if path != None:
        
        fig.savefig(path, dpi = 200)
    
    fig.show()
    
    return Bias
    
    
# =============================================================================
# def bias2(sample):
#     
#     """
#     sample: A array with n repeats from x's distribution
#     
#     return: The Variance, Bias and Mse from X estimators
#     """
#     
#     moms_ = sample.apply(mom2)
#     
#     bootst = sample.apply(lambda x: Bootst_2(x, 1000))
#     
#     r_bootst = bootst.loc[moms_.index,:]
#     
#     bias = (moms_-r_bootst)
#     
#     bias = pd.DataFrame(dict(
#         Bgamma = bias.iloc[:3,:].apply(lambda x: np.linalg.norm(x, axis = 0)**2),
#         Bln = bias.iloc[3:6,:].apply(lambda x: np.linalg.norm(x, axis = 0)**2),
#         Bgum = bias.iloc[6:9,:].apply(lambda x: np.linalg.norm(x, axis = 0)**2)
#         )).T
# 
#     var = bootst.loc[bootst.index[9:],:]
#     
#     var_ = var.copy()
#     
#     var_.index = bias.index
#     
#     mse = bias+var_
#     
#     mse.index = ["Mgamma", "Mln", "Mgum"]
#     
#     return dict(bias = bias, var = var, mse = mse)
# =============================================================================

def fit(sample, parameters, graph = False):
    
    if graph:
    
        fig, ax = plt.subplots(figsize=(6,6))
    
        qq = ss.probplot(sample, sparams=(parameters[1]**(1/2), 0, np.exp(parameters[0])),
                dist = ss.lognorm, plot = ax)
    
    #    np.random.seed(1914)
    
        ad = ss.anderson_ksamp([qq[0][1], qq[0][0]
                            ]).pvalue
    
        ax.set_title("AD-Pvalue: %.4f"%(ad))
    
        fig.suptitle(f"{sample.name}")
    
        ax.set_xlabel("Cuantil teórico")
    
        ax.set_ylabel("Cuantil ordenado")
    
        fig.tight_layout()
    
        fig.show()
    
    elif parameters.shape[0]>2:
        
        qq_ln = ss.probplot(sample, sparams=(parameters["ln_b"]**(1/2), 0, np.exp(parameters["ln_m"])),
                dist = ss.lognorm)
        
        qq_gam = ss.probplot(sample, sparams=(parameters["gamma_a"], 0, parameters["gamma_b"]),
                dist = ss.gamma)
        
        qq_gum = ss.probplot(sample, sparams=(0, parameters["gum_m"], parameters["gum_b"]),
                dist = ss.genextreme)
        
        ad = dict(ad_ln = ss.anderson_ksamp([qq_ln[0][1], qq_ln[0][0]
                            ]).pvalue,
                  ad_gamma = ss.anderson_ksamp([qq_gam[0][1], qq_gam[0][0]
                                                ]).pvalue, 
                  ad_gum = ss.anderson_ksamp([qq_gum[0][1], qq_gum[0][0]
                                                ]).pvalue)
        
        ad = pd.Series(ad)
        
    else:
        
        raise Exception("Parameters not be comparable")
    
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
#%% argmax

def kendall_tau(sample, parameter, cfunction):
    """
    parameter: Possible parameter from random sample
    sample: U and V array of values from any distribution
    cfunction: A possible copula function
    return: A Fastest estimation of kendall tau
    """
    
    return 4*np.mean(cfunction(sample, parameter))-1

def k_t(cfunction, parameter, nint = 1000):
    """
    parameter: Possible parameter from random sample
    cfunction: A possible copula function
    nint: n of iteration or divisions for the sample
    return: A approximation of kendall tau integrate
    """
    
    return 4*CopulaIM(cfunction, parameter, nint)-1

def r_s(cfunction, parameter, nint=1000):
    
    """
    parameter: Possible parameter from random sample
    cfunction: A possible copula function
    nint: n of iteration or divisions for the sample
    return: A approximation of kendall tau integrate
    """
    
    return 12*CopulaIM(cfunction, parameter, nint)-3


def k_t2(cfunction, parameter, nint = 1000):
    """
    parameter: Possible parameter from random sample
    cfunction: A possible copula function
    nint: n of iteration or divisions for the sample
    return: A approximation of kendall tau integrate
    """
    
    return 4*CopulaIM2(cfunction, parameter, nint)-1

def r_s2(cfunction, parameter, nint=1000):
    
    """
    parameter: Possible parameter from random sample
    cfunction: A possible copula function
    nint: n of iteration or divisions for the sample
    return: A approximation of kendall tau integrate
    """
    
    return 12*CopulaIM2(cfunction, parameter, nint)-3

def argmax(parameter, n, cfunction, kendalltau):
    
    """
    parameter: Possible parameter from random sample
    sample: U and V array of values from any distribution
    cfunction: A possible copula function
    kendaltau: Concordance non parameter estimation
    """   
    return k_t(cfunction, parameter)-kendalltau

def kv(key, kt):
    """
    key: Copula name
    kt: empirical kendall tau
    
    return: function and kendall bound, and
    parameter bound
    """
    
    if key == "clayton":
        
        from Copulas.clayton import Clayton
        
        function = Clayton
        
        lim = [-1, 1]
        
        plim = [-1, 50]
        
        #0, 1
    
    elif key == "gumbel_hougaard":
        
        from Copulas.gumbel_hougaard import Gumbel_Hougaard
    
        function = Gumbel_Hougaard
        
        lim = [0, 1]
        
        plim = [1, 50]
        
        #0, 1
    
    elif key == "gumbel_barnett":
        
        from Copulas.gumbel_barnett import Gumbel_Barnett

        function = Gumbel_Barnett
    
        lim = [0, -.361]
        
        plim = [0,1]
        
        #0, -.3612
        
    elif key == "fgm":
        
        from Copulas.fgm import FGM
        
        function = FGM
        
        lim = [-.222, .222]
        
        plim = [-1,1]
        
        # -0.222, 0.222
        
    elif key == "frank":
        
        from Copulas.frank import Frank
        
        function = Frank
        
        lim = [-1, 1]
        
        plim = [-36, 36]
        
        # -1, 1
        
    else:
        
        raise Exception("The function is out of study")
    
    return (function, lim, plim)

def maxkt(XY, n, function, kt, plim, lim, argmax = argmax):
    """
    n: sample size
    function: Copula function
    kt: Empirical kendall tau
    plim: parameter bound
    lim: kendall tau bound
    argmax: kendall maximun function
    
    return: alpha and a0
    
    """
   # print(function)
  
    if (min(lim) < kt < max(lim)):
        
        a0 = so.root_scalar(argmax,
                            args=(n, function, kt),
                            bracket=(plim[lim.index(min(lim))],
                                plim[lim.index(max(lim))])).root
        
        alpha = max_ll(XY, function, a0)[0]
        
    elif abs(min(lim)-kt)<=0.05:
        
        alpha = plim[lim.index(min(lim))]
        
        a0 = np.nan
        
    elif abs(max(lim)-kt)<=0.05:
        
        alpha = plim[lim.index(max(lim))]
        
        a0 = np.nan
    
    elif not (min(lim) < kt < max(lim)):
        
        raise Exception("Kendall tau out of limit")
    
    
    return (a0, alpha)
    
    
#%% T test

def T(XY, u, v, function ,alpha):
    
    qn = Qn(XY, u, v, Cn)
    
    q = Q(u, v, function, [alpha])
    
    qn_ = qn[(abs(q)<=1)&(abs(qn)<=1)]
    
    q_ = q[(abs(q)<=1)&(abs(qn)<=1)]
   # fitdist(np.sqrt(len(q_))*(qn_-q_))
    t_c = ((q_-qn_)**2).sum()
    
    return t_c

def Ttest(sample, key, n_bootst=1000, RD = False):
    
    """
    sample: DataFrame bivariate random sample
    key: Possible copula
    n_bootst: Bootstrap sample from real values of statistic
    
    return: Sample statistic, Pvalue of statistic,
    Kendall's Tau, Copula Parameter
    
    """
    
    sample = np.array(u_v(sample))
    
    #print(sample)
    
    return_ = []
    
    m = n_bootst
    
    n = sample.shape[1]
    
    kt = ss.kendalltau(*sample)[0]
    
    function, lim, plim = kv(key, kt)
        
    XY = pd.DataFrame(dict(x=sample[0], y=sample[1]))
    
    #print(sample)
    #print(kt)
    a0, alpha = maxkt(XY, n, function, kt, plim, lim)
    #print(a0)
    #print(alpha)
    
    #print(n)
    
    t_gen(int(m*1.1), n, key, alpha, return_)
    
    t_sam = np.array(return_).ravel()[:m]
    
    t_c = T(XY, sample[0], sample[1], function, alpha)
    
    Ptc = (t_sam>=t_c).sum()/m
    
    if RD:
        
        k = moms(t_sam)
        
        ln = ss.lognorm(k["ln_b"]**(1/2), 0, np.exp(k["ln_m"]))

        gum = ss.genextreme(0, k["gum_m"], k["gum_b"])        
        
        return pd.DataFrame(dict(Tc = t_c,
                             PvalueTc = Ptc,
                             Pvalueln = ln.sf(t_c),
                             Pvaluegum = gum.sf(t_c),
                             EKT = kt,
                             itaualpha = a0,
                             mlealpha = alpha,
                             PKT = k_t(function, alpha, 1000).round(4)),
                        index = ["Results"])
    
    else:
        
        return pd.DataFrame(dict(Tc = t_c,
                             PvalueTc = Ptc,
                             EKT = kt,
                             itaualpha = a0,
                             mlealpha = alpha,
                             PKT = k_t(function, alpha, 1000).round(4)),
                        index = ["Results"])
    
    
    
    

def Pow_Conf(sample_, tkey, m_resamples=1000, RD = False, key = None,
             calpha = 0.05, n_bootst = 1000):
    
    sample_ = np.array(u_v(sample_))
    
    #print(sample)
    
    return_ = []
    
    m = m_resamples
    
    n = sample_.shape[1]
    
    # Estimation of parameters (It's slower)
    
    kt = ss.kendalltau(*sample_)[0]
    
    function, lim, plim = kv(tkey, kt)
     
    #print(function)
    
    XY = pd.DataFrame(dict(x=sample_[0], y=sample_[1]))
    
    #print(sample)
    #print(kt)
    a0, alpha = maxkt(XY, n, function, kt, plim, lim)
    
    # Simulate T (It's slower)
    
    t_gen(int(m*1.1), n, tkey, alpha, return_)
    
    t_sam = np.array(return_).ravel()[:m]
    
    # Take T simulations and do resamples of T
    
    t_bootst = Bootsttrap(t_sam, k=n_bootst)
    
    # If RD is true, make RD simulate and take it for RD test
    # Take the parameter and do Simulate of new values of U y V
    
    if key == None:
        
        uv = list(map(lambda x: pd.DataFrame(sample(n, tkey, 
                                                     alpha),
                                              index = ["x","y"]).T
                        ,range(n_bootst)))
        
        t_c = list(map(lambda x: T(x, x.x, x.y, function, alpha),
                       uv))
        
    
        nbTest = (t_bootst>np.array(t_c)).mean(axis=0)
    
    
        return (nbTest>calpha).mean()
        
    # For the power evalue P(Tr>trc)<alpha
    
    elif key !=None:
        
        kt = ss.kendalltau(*sample_)[0]
        
        function, lim, plim = kv(key, kt)
            
        XY = pd.DataFrame(dict(x=sample_[0], y=sample_[1]))
        
        #print(sample)
        #print(kt)
        a0, alpha = maxkt(XY, n, function, kt, plim, lim)
        
        uv = list(map(lambda x: pd.DataFrame(sample(n, key, 
                                                     alpha),
                                              index = ["x","y"]).T
                        ,range(n_bootst)))
        
        t_c = list(map(lambda x: T(x, x.x, x.y, function, alpha),
                       uv))
    
        nbTest = (t_bootst>np.array(t_c)).mean(axis=0)
    
    
        return (nbTest<=calpha).mean()
        
    # For the confidence evalue P(Tr>trc)>=alpha
    
    #Resultados clayton/frank n=1000, theta = 8, conf = .946 pow = 0.388
