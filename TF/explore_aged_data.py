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

# load battery data
data_RW_all = getDischargeMultipleBatteries()
data_RW = {1: data_RW_all[1]} # only battery 1
max_idx_to_use = 100
max_size = np.max([ v[0,0].shape[0] for k,v in data_RW.items() ])

dt = np.diff(data_RW[1][2,0])[1]

inputs = None
target = None
for k,v in data_RW.items():
    for i,d in enumerate(v[1,:][:max_idx_to_use]):
        prep_inp = np.full(max_size, np.nan)
        prep_target = np.full(max_size, np.nan)
        prep_inp[:len(d)] = d
        prep_target[:len(v[0,:][i])] = v[0,:][i]
        if inputs is None:
            inputs = prep_inp
            target = prep_target
        else:
            inputs = np.vstack([inputs, prep_inp])
            target = np.vstack([target, prep_target])

inputs = inputs[:,:,np.newaxis]
time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]

# move timesteps with earlier EOD
EOD = 3.2

inputs_shiffed = inputs.copy()
target_shiffed = target.copy()
reach_EOD = np.ones(BATCH_SIZE, dtype=int) * time_window_size
for row in np.argwhere((target<EOD) | (np.isnan(target))):
    if reach_EOD[row[0]]>row[1]:
        reach_EOD[row[0]]=row[1]
        inputs_shiffed[row[0],:,0] = np.zeros(time_window_size)
        inputs_shiffed[row[0],:,0][time_window_size-row[1]:] = inputs[row[0],:,0][:row[1]]
        target_shiffed[row[0]] = np.ones(time_window_size) * target[row[0]][0]
        target_shiffed[row[0]][time_window_size-row[1]:] = target[row[0]][:row[1]]


SIMULATION_OVER_STEPS = 200
inputs_shiffed = np.hstack([inputs_shiffed, inputs_shiffed[:, -SIMULATION_OVER_STEPS:]])
inputs = np.hstack([inputs, inputs[:, -SIMULATION_OVER_STEPS:]])
time_window_size = inputs_shiffed.shape[1]


checkpoint_filepath = './training/cp_mlp_save4.ckpt'

base_model = get_model(batch_input_shape=(36,1,1), dt=dt, mlp=True, share_q_r=False)
base_model.load_weights(checkpoint_filepath)
base_weights = base_model.get_weights()

weights = base_weights.copy()


range_size = 100
total_time = 10000
q_max_range = np.linspace(0.5e4,1.5e4,range_size)
# R_0_range = np.linspace(0.4e-1,1.5e-1,range_size)
R_0_range = np.linspace(0.5e-1,4e-1,range_size)
input_range = np.arange(0,total_time,dt)
X, Y = np.meshgrid(q_max_range, R_0_range)
Z = np.zeros((range_size,range_size))

inputs = np.ones((range_size, input_range.shape[0], 1))

model_eval = get_model(batch_input_shape=(range_size,input_range.shape[0],1), dt=dt, mlp=True, share_q_r=False, q_max_base=1.0, R_0_base=1.0)


scamap = plt.cm.ScalarMappable(cmap='coolwarm')
# fcolors = scamap.to_rgba(q_max_range)
fcolors = scamap.to_rgba(R_0_range)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in np.arange(range_size, dtype=int)[::10]:
    weights[0] = X[i]
    weights[1] = Y[i]
    model_eval.set_weights(weights)
    pred = model_eval.predict(inputs)[:,:,0]

    for j in np.arange(range_size, dtype=int)[::10]:
        # ax.plot(input_range, pred[j,:], Y[i][j], color=fcolors[j], zdir='y')
        z_idx = pred[j,:]>=3.2
        ax.plot(input_range[z_idx], pred[j,z_idx], X[i][j], color=fcolors[i], zdir='y')

    reach_EOD_pred = np.ones(range_size, dtype=int) * input_range.shape[0]
    for row in np.argwhere(pred<EOD):
        if reach_EOD_pred[row[0]]>row[1]:
            reach_EOD_pred[row[0]]=row[1]
    Z[i] = reach_EOD_pred

ax.set_zlim(3.2, 4.2)
ax.set_xlabel('Time (S)')
ax.set_ylabel(r'$q_{max}$')
ax.set_zlabel('Voltage (V)')

norm = matplotlib.colors.Normalize(vmin=np.min(R_0_range), vmax=np.max(R_0_range))
cb1 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm), ax=ax)
cb1.set_label(r'$R_0$')

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
