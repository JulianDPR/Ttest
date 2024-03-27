import numpy as np
import pandas as pd
import scipy.stats as ss
import time as t
from tools import ranks
import matplotlib.pyplot as plt

def Cn_(XY, u, v):
    n = len(XY)
    R = np.argsort(XY[:, 0]) + 1
    S = np.argsort(XY[:, 1]) + 1
    
    count = np.sum((R / (n + 1) <= u[:, None]) & (S / (n + 1) <= v[:, None]), axis=0)
    
    cn = count / n
    
    return cn

def Cn(XY, u, v):
    
    n, R, S = ranks(XY)
    cn = np.sum((R / (n + 1) <= u[:, None]) & (S / (n + 1) <= v[:, None]), axis=1)/n
    
    return cn

def Cn__(XY, u, v):
    
    n, R, S = ranks(XY)
    cn = ((R / (n + 1) <= u) * (S / (n + 1) <= v)).sum()/n
    
    return cn



n = 1000
u = np.array([0.97, 0.54, 0.73, 0.17, 0.32])
v = np.array([0.21, 0.18, 0.55, 0.36, 0.82])
XY = pd.DataFrame(dict(x=u, y=v)).to_numpy()
u,v = np.meshgrid(*[np.linspace(1/5, 1, 5),np.linspace(1/5, 1, 5)])

print(np.matrix(list(map(lambda x,y: Cn(pd.DataFrame(XY, columns = ["x","y"]),x, y), u, v))))

print(np.matrix(list(map(lambda x,y: np.vectorize(Cn__, excluded=[0])(pd.DataFrame(XY, columns = ["x","y"]),x,y), u, v))))


#%%

from tools import Bootst
from tools import fitdist
from tools import max_ll
from tools import log_likelihood
import scipy.optimize as so
import scipy.stats as ss
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/Bienvenido/Desktop/TG/Imag/gumbel_hougaard/muestra.xlsx"

df = pd.read_excel(path, sheet_name = "strong")

par = max_ll(df, ss.genextreme, [0.94, 0.23], log_likelihood)

df["ranks"] = df.sort_values(by=0).reset_index().sort_values(by="index").reset_index()["level_0"]+1

df["u"] = df["ranks"]/df.shape[0]

df["f"] = ss.genextreme(0, *par).ppf(df["u"])

df.columns = ["x"]+list(df.columns[1:])

fig, ax = plt.subplots(figsize=(10,10))

sns.regplot(df, x="x", y="f", line_kws = dict(color="darkred", alpha=1, linewidth=100),
            scatter_kws=dict(color="darkcyan", alpha=0.8),
            marker=".", ax=ax, ci = 95)

ax.grid()

#%%
from scipy.optimize import ridder

# Define la función que quieres minimizar con argumentos adicionales
def funcion_a_minimizar(x, a, b):
    return a * x ** 3 + b * x

# Define los valores iniciales para los argumentos
a_valores = [1, 2, 3]  # Varios valores para 'a'
b_valor = 5  # Un único valor para 'b'

# Función auxiliar para minimizar con argumentos adicionales fijos
def funcion_auxiliar(x):
    return funcion_a_minimizar(x, b=b_valor, a=a_valor)

# Llama a la función ridder para encontrar el mínimo
resultados = []
for a_valor in a_valores:
    resultado = ridder(funcion_auxiliar, -10, 10)
    resultados.append(resultado)

print("Los mínimos de la función para diferentes valores de 'a' son:", resultados)





