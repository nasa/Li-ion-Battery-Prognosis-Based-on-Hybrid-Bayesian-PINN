# %% imports and const
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

F = 96487.0
# %% funcs
V_INT_k = lambda x,i: (2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i)

V_INT = lambda x,A: np.dot(A, np.array([V_INT_k(x,i) for i in range(len(A))])) / F

def Ai(A,i,a):
    A[i]=a
    return A
# %%
Ap = np.array([
    -31593.7,
    0.106747,
    24606.4,
    -78561.9,
    13317.9,
    307387.0,
    84916.1,
    -1.07469e+06,
    2285.04,
    990894.0,
    283920,
    -161513,
    -469218
])
# %% gen data
PARAM_i = 0  # which param to explore

param_mag = max(10.0, 10.0**np.ceil(np.log10(np.abs(Ap[PARAM_i]))))

X = np.linspace(0.0,1.0,100)
Api = np.linspace(-1.0,1.0,100)*param_mag
Xm,Am = np.meshgrid(X,Api)
x = np.ravel(Xm)
api = np.ravel(Am)
V_INT_p = np.array([V_INT(x, Ai(Ap.copy(),PARAM_i,api[i])) for i,x in enumerate(x)]).reshape(Xm.shape)
V_INT_defaults = np.array([V_INT(x, Ap) for i,x in enumerate(X)])

# %% plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Xm, Am, V_INT_p, cmap=cm.coolwarm)
ax.plot(X,np.ones_like(X)*Ap[PARAM_i],V_INT_defaults)

ax.set_xlabel('x - mole frac')
ax.set_ylabel(r'$A_{{p,{:}}}$'.format(PARAM_i))
ax.set_zlabel('$V_{INT,p}$')

plt.show()

# %%
