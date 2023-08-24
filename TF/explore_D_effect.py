import numpy as np
import math
from time import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

from model import get_model

from battery_data import getDischargeMultipleBatteries

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)
dt=10.0

# checkpoint_filepath = './training/cp_mlp_save4.ckpt'

# base_model = get_model(batch_input_shape=(36,1,1), dt=dt, mlp=True, share_q_r=False)
# base_model.load_weights(checkpoint_filepath)
# base_weights = base_model.get_weights()

# weights = base_weights.copy()

range_size = 100
total_time = 10000
q_max_range = np.linspace(0.5e4,1.5e4,range_size)
# R_0_range = np.linspace(0.4e-1,1.5e-1,range_size)
R_0_range = np.linspace(0.5e-1,4e-1,range_size)
D_range = np.linspace(7e6,7e7,range_size)

input_range = np.arange(0,total_time,dt)

R_0 = 1e-1*np.ones(range_size)

# X, Y = np.meshgrid(q_max_range, R_0_range)
X, Y = np.meshgrid(q_max_range, D_range)
Z = np.zeros((range_size,range_size))

inputs = np.ones((range_size, input_range.shape[0], 1))

model_eval = get_model(batch_input_shape=(range_size,input_range.shape[0],1), dt=dt, mlp=True, share_q_r=False, q_max_base=1.0, R_0_base=1.0, D_trainable=True)

cell = model_eval.layers[0].cell
cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))
MLPn_weigths = np.load('./training/MLPn_best_weights.npy',allow_pickle=True)
cell.MLPn.set_weights([MLPn_weigths[:1], MLPn_weigths[1]])

weights = model_eval.get_weights().copy()

scamap = plt.cm.ScalarMappable(cmap='coolwarm')
# fcolors = scamap.to_rgba(q_max_range)
# fcolors = scamap.to_rgba(R_0_range)
fcolors = scamap.to_rgba(D_range)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in np.arange(range_size, dtype=int)[::10]:
    weights[0] = X[i]
    # weights[1] = Y[i]
    weights[1] = R_0
    weights[2] = Y[i][0]
    model_eval.set_weights(weights)
    pred = model_eval.predict(inputs)[:,:,0]

    for j in np.arange(range_size, dtype=int)[::10]:
        # ax.plot(input_range, pred[j,:], Y[i][j], color=fcolors[j], zdir='y')
        z_idx = pred[j,:]>=3.2
        ax.plot(input_range[z_idx][4:], pred[j,z_idx][4:], X[i][j], color=fcolors[i], zdir='y')

ax.set_zlim(3.2, 4.2)
ax.set_xlabel('Time (S)')
ax.set_ylabel(r'$q_{max}$')
ax.set_zlabel('Voltage (V)')

norm = matplotlib.colors.Normalize(vmin=np.min(D_range), vmax=np.max(D_range))
cb1 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), ax=ax)
# cb1.set_label(r'$R_0$')
cb1.set_label('D')
# cb1.ax.set_yticklabels(["{:.2e}".format(i) for i in cb1.get_ticks()])

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
