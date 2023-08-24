import numpy as np
import math
from time import time
import argparse
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from model import get_model

from battery_data import getDischargeMultipleBatteries

matplotlib.rc('font', size=14)

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)

parser = argparse.ArgumentParser()
parser.add_argument("--save", default=False, action="store_true" , help="Save plots and results")
parser.add_argument("--mse", default=False, action="store_true" , help="Generate mse for each sample")
args = parser.parse_args()

# load all battery data
BATTERY_NUM = '1to8'
data = np.load('./training/input_data_refer_disc_batt_1to8.npy', allow_pickle=True).item()
# BATTERY_NUM = 3
# data = np.load('./training/input_data_refer_disc_batt_{}.npy'.format(BATTERY_NUM), allow_pickle=True).item()

inputs = data['inputs']
target = data['target']
inputs_time = data['time']

time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]
dt = inputs_time[0,1]-inputs_time[0,0]
max_size = inputs.shape[1]


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

val_idx = np.linspace(0,48,8,dtype=int)
train_idx = [i for i in np.arange(0,48) if i not in val_idx]

inputs_shiffed_all = inputs_shiffed
inputs_all = inputs
target_all = target
target_shiffed_all = target_shiffed
reach_EOD_all = reach_EOD
# inputs_shiffed = inputs_shiffed[train_idx,:,:]
# inputs = inputs[train_idx,:,:]
# target = target[train_idx,:]
# target_shiffed = target_shiffed[train_idx,:]
# reach_EOD = reach_EOD[train_idx]

q_max_base = 1.0e3
R_0_base = 1.0e1

model_eval = get_model(batch_input_shape=(1,time_window_size-SIMULATION_OVER_STEPS,1), dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model_eval.compile(optimizer='adam', loss="mse", metrics=["mae"])
model = get_model(batch_input_shape=inputs.shape, dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model.compile(optimizer='adam', loss="mse", metrics=["mae"])


model.layers[0].cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))
# MLPn_weigths = np.load('./training/MLPn_best_weights.npy',allow_pickle=True)
# model.layers[0].cell.MLPn.set_weights([MLPn_weigths[:1], MLPn_weigths[1]])

xi = np.linspace(0.0,1.0,100)
fig = plt.figure('MLPp')
plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]), color='gray')

# fig = plt.figure('MLPn')
# plt.plot(xi, model.layers[0].cell.MLPn(xi[:,np.newaxis]), color='gray')

checkpoint_filepath = './training/cp_mlp_aged_batt_1to8.ckpt'
# checkpoint_filepath = './training/cp_mlp_aged_batt_{}.ckpt'.format(BATTERY_NUM)
# checkpoint_filepath = './training/cp_mlp_aged_batt_3-WITH-MLP.ckpt'
model.load_weights(checkpoint_filepath)
weights = model.get_weights()

if args.save:
    np.save('./training/model_weights_refer_disc_batt_{}.npy'.format(BATTERY_NUM), weights)


F = 96487.0
V_INT_k = lambda x,i: (2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i)
V_INT = lambda x,A: np.dot(A, np.array([V_INT_k(x,i) for i in range(len(A))])) / F
def Ai(A,i,a):
    A[i]=a
    return A
# reference Redlich-Kister coeff vals
Ap = np.array([-31593.7, 0.106747, 24606.4, -78561.9, 13317.9, 307387.0, 84916.1, -1.07469e+06, 2285.04, 990894.0, 283920, -161513, -469218])

Vint = np.array([V_INT(x,Ap) for x in xi])
fig = plt.figure('MLPp')
plt.plot(xi,Vint, label='Redlich V_INT')
plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]), label='MLP V_INT')
plt.grid()
plt.legend()

if args.save:
    fig.savefig('./figures/Vint_p_mlp_aged_batt_{}.png'.format(BATTERY_NUM))

fig = plt.figure('MLPn')
plt.plot(xi, model.layers[0].cell.MLPn(xi[:,np.newaxis]))
plt.grid()

if args.save:
    fig.savefig('./figures/Vint_n_mlp_aged_batt_{}.png'.format(BATTERY_NUM))


pred_shiffed = model.predict(inputs_shiffed)[:,:,0]
# print('Model Eval [mse,mae]:', model_eval.evaluate(inputs_shiffed[:,:-SIMULATION_OVER_STEPS,:], target_shiffed))

# pred = model.predict(inputs)
pred = np.full((inputs.shape[0],inputs.shape[1]), np.nan)
for i in range(pred.shape[0]):
    pred[i, :(reach_EOD[i]+SIMULATION_OVER_STEPS)] = pred_shiffed[i, (max_size - reach_EOD[i]):]


if args.mse:
    mse = np.zeros(inputs.shape[0])
    weights_eval = weights.copy()
    for i in range(inputs.shape[0]):
        weights_eval[0] = np.reshape(weights[0][i], (1,))
        weights_eval[1] = np.reshape(weights[1][i], (1,))
        # weights_eval[2] = np.reshape(weights[2][i], (1,))
        # weights_eval[3] = np.reshape(weights[3][i], (1,))
        model_eval.set_weights(weights_eval)
        mse[i] = model_eval.evaluate(inputs_shiffed[i,:target_shiffed.shape[1],:][np.newaxis,:,:], target_shiffed[i,:][np.newaxis,:,np.newaxis])[0]
        print("MSE[{}]: {}".format(i, mse[i]))

    print("")
    print("AVG MSE:, ", mse.mean())

    fig = plt.figure()
    plt.hist(mse)
    plt.xlabel(r'mse')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    if args.save:
        fig.savefig('./figures/mse_dist_aged_batt_{}.png'.format(BATTERY_NUM))


fig = plt.figure()
plt.hist(weights[0]*model.layers[0].cell.qMaxBASE.numpy())
plt.xlabel(r'$q_{max}$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

if args.save:
    fig.savefig('./figures/q_max_dist_aged_batt_{}.png'.format(BATTERY_NUM))

fig = plt.figure()
plt.hist(weights[1]*model.layers[0].cell.RoBASE.numpy())
plt.xlabel(r'$R_0$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

if args.save:
    fig.savefig('./figures/R_0_dist_aged_batt_{}.png'.format(BATTERY_NUM))


fig = plt.figure('q_max_over_time')
plt.plot(inputs_time[:,0]/3600, weights[0]*model.layers[0].cell.qMaxBASE.numpy(), 'o', label='Ref. Dis.')
# plt.xlabel(r'Sample')
plt.xlabel('Time (h)')
plt.ylabel(r'$q_{max}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()
# plt.legend()

if args.save:
    fig.savefig('./figures/q_max_time_aged_batt_{}.png'.format(BATTERY_NUM))

fig = plt.figure('R_0_over_time')
plt.plot(inputs_time[:,0]/3600, weights[1]*model.layers[0].cell.RoBASE.numpy(), 'o', label='Ref. Dis.')
# plt.xlabel(r'Sample')
plt.xlabel('Time (h)')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()
# plt.legend()

if args.save:
    fig.savefig('./figures/R_0_time_aged_batt_{}.png'.format(BATTERY_NUM))

data_to_save = {
    'time': inputs_time[:,0],
    'q_max': weights[0]*model.layers[0].cell.qMaxBASE.numpy(),
    'R_0': weights[1]*model.layers[0].cell.RoBASE.numpy()
}

if args.save:
    np.save('./training/q_max_R_0_aged_batt_{}.npy'.format(BATTERY_NUM),  data_to_save)


time_axis = np.arange(time_window_size) * dt
cmap = matplotlib.cm.get_cmap('Spectral')

fig = plt.figure()

# plt.subplot(311)
# for i in range(inputs.shape[0]):
#     plt.plot(time_axis, inputs[i,:,0])
# plt.ylabel('Current (A)')
# plt.grid()

plt.subplot(211)
for i in range(pred_shiffed.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], target_shiffed[i,:], color='gray')
for i in range(pred_shiffed.shape[0]):
    idx_end = len(time_axis)
    idx = np.argwhere(pred_shiffed[i,:]<EOD)
    if len(idx):
        idx_end = idx[0][0]
    plt.plot(time_axis[:idx_end], pred_shiffed[i,:idx_end])
plt.ylabel('Voltage (V)')
plt.grid()

plt.subplot(212)
for i in range(pred.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], target[i,:], color='gray')
for i in range(pred.shape[0]):
    idx_end = len(time_axis)
    idx = np.argwhere(pred[i,:]<EOD)
    if len(idx):
        idx_end = idx[0][0]
    plt.plot(time_axis[:idx_end], pred[i,:idx_end])
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
# plt.ylim([3.0,4.2])
plt.grid()

if args.save:
    fig.savefig('./figures/V_pred_aged_batt_{}.png'.format(BATTERY_NUM))


fig = plt.figure()

plt.subplot(211)
for i in range(pred_shiffed.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], pred_shiffed[i,:-SIMULATION_OVER_STEPS] / target_shiffed[i,:])

plt.ylabel('Pred / Actual Ratio (V)')
plt.grid()

plt.subplot(212)
for i in range(pred.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], pred[i,:-SIMULATION_OVER_STEPS] / target[i,:])

plt.ylabel('Pred / Actual Ratio (V)')
# plt.ylim([3.0,4.2])
plt.grid()

plt.xlabel('Time (s)')

if args.save:
    fig.savefig('./figures/pred_ratio_aged_batt_{}.png'.format(BATTERY_NUM))


reach_EOD_pred = np.ones(inputs.shape[0], dtype=int) * time_window_size
for row in np.argwhere(pred<EOD):
    if reach_EOD_pred[row[0]]>row[1]:
        reach_EOD_pred[row[0]]=row[1]

fig = plt.figure()
EOD_range = [min(np.min(reach_EOD*dt),np.min(reach_EOD_pred*dt)),max(np.max(reach_EOD*dt),np.max(reach_EOD_pred*dt))]
plt.plot(EOD_range, EOD_range, '--k')
plt.plot(reach_EOD*dt, reach_EOD_pred*dt, '.')
plt.ylabel("Predicted EOD (s)")
plt.xlabel("Actual EOD (s)")
plt.xlim(EOD_range)
plt.ylim(EOD_range)
plt.grid()

if args.save:
    fig.savefig('./figures/EOD_pred_aged_batt_{}.png'.format(BATTERY_NUM))


plt.show()
