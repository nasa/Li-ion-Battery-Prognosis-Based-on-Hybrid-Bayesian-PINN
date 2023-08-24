import numpy as np
import math
import argparse
from time import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from model import get_model

# from battery_data import getDischargeMultipleBatteries

matplotlib.rc('font', size=14)

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)

parser = argparse.ArgumentParser()
parser.add_argument("--save", default=False, action="store_true" , help="Save plots and results")
args = parser.parse_args()

# %% load all battery data
# data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type='discharge (random walk)')
# data_RW_all = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type='discharge (random walk)')

# %% process data
# data_RW_all = getDischargeMultipleBatteries()
BATTERY_NUM = 3
data = np.load('./training/input_data_rw_disc_batt_{}.npy'.format(BATTERY_NUM), allow_pickle=True).item()

inputs = data['inputs']
target = data['target']
inputs_time = data['time']

# num_seq = 10  # number of sequences of rw discharge of each battery to include
# # num_batteries = 8  # up until battery to include (9 to 12 has diff discharge profile)

# max_size = 0
# inputs = []
# inputs_time = []
# target = []

# for k,rw_data in data_RW.items():
#     # if k>num_batteries:
#     #     continue
#     # rw_data = data_RW[1]
#     time = np.hstack([rw_data[2][i] for i in range(len(rw_data[2]))])
#     time = time - time[0]
#     current_inputs = np.hstack([rw_data[1][i] for i in range(len(rw_data[1]))])
#     voltage_target = np.hstack([rw_data[0][i] for i in range(len(rw_data[0]))])

#     last_idx = 0
#     seq_durations = np.diff([0]+list(np.argwhere(np.diff(time)>10)[:,0]+1))
    
#     # for curr_duration in seq_durations[:num_seq]:
#     for curr_duration in seq_durations:
#         if curr_duration>max_size:
#             max_size = curr_duration
#         curr_idx = last_idx + curr_duration
#         inputs.append(current_inputs[last_idx:curr_idx])
#         inputs_time.append(time[last_idx:curr_idx])
#         target.append(voltage_target[last_idx:curr_idx])
#         last_idx = curr_idx

# # add nan to end of seq to have all seq in same size
# for i in range(len(inputs)):
#     prep_inputs = np.full(max_size, np.nan)
#     prep_target = np.full(max_size, np.nan)
#     prep_inputs_time = np.full(max_size, np.nan)
#     prep_inputs[:len(inputs[i])] = inputs[i]
#     prep_target[:len(target[i])] = target[i]
#     prep_inputs_time[:len(inputs_time[i])] = inputs_time[i]
#     inputs[i] = prep_inputs
#     target[i] = prep_target
#     inputs_time[i] = prep_inputs_time

# inputs = np.vstack(inputs)[:,:,np.newaxis]
# target = np.vstack(target)
# inputs_time = np.vstack(inputs_time)

# select every other 3 samples
inputs = inputs[::3,:,:]
target = target[::3,:]
inputs_time = inputs_time[::3,:]

max_size = inputs.shape[1]


time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]
dt = inputs_time[0,1]-inputs_time[0,0]
# dt=10.0

# %% calc EOD
# move timesteps with earlier EOD
EOD = 3.2

inputs_shiffed = inputs.copy()
target_shiffed = target.copy()
reach_EOD = np.ones(BATCH_SIZE, dtype=int) * time_window_size
for row in np.argwhere((target<EOD) | (np.isnan(target))):
    if reach_EOD[row[0]]>row[1]:
        reach_EOD[row[0]]=row[1]
        row_1 = row[1]
        if ~np.isnan(inputs[row[0],:,0][row[1]]):
            row_1 = row[1] + 1
        inputs_shiffed[row[0],:,0] = np.zeros(time_window_size)
        inputs_shiffed[row[0],:,0][time_window_size-row_1:] = inputs[row[0],:,0][:row_1]
        target_shiffed[row[0]] = np.ones(time_window_size) * target[row[0]][0]
        target_shiffed[row[0]][time_window_size-row_1:] = target[row[0]][:row_1]

# %%

# limit to first rw only
inputs_shiffed_all = inputs_shiffed
target_shiffed_all = target_shiffed
inputs_all = inputs
target_all = target
reach_EOD_all = reach_EOD

# inputs_shiffed = inputs_shiffed[:,::10,:]
# target_shiffed = target_shiffed[:,::10]
# inputs = inputs[:,::10,:]
# target = target[:,::10]
# dt=10.0

reach_EOD = reach_EOD//10
max_size = inputs.shape[1]

time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]

SIMULATION_OVER_STEPS = 300
inputs_shiffed = np.hstack([inputs_shiffed, (np.ones((BATCH_SIZE, SIMULATION_OVER_STEPS)) * inputs_shiffed[:, -1, 0][:,np.newaxis])[:,:,np.newaxis]])
inputs = np.hstack([inputs, (np.ones((BATCH_SIZE, SIMULATION_OVER_STEPS)) * inputs[:, -1, 0][:,np.newaxis])[:,:,np.newaxis]])
time_window_size = inputs_shiffed.shape[1]

# val_idx = np.linspace(0,35,6,dtype=int)
# train_idx = [i for i in np.arange(0,36) if i not in val_idx]

# inputs_shiffed_all = inputs_shiffed
# inputs_all = inputs
# target_all = target
# target_shiffed_all = target_shiffed
# inputs_shiffed = inputs_shiffed[train_idx,:,:]
# inputs = inputs[train_idx,:,:]
# target = target[train_idx,:]
# target_shiffed = target_shiffed[train_idx,:]
# reach_EOD_all = reach_EOD
# reach_EOD = reach_EOD[train_idx]

# checkpoint_filepath = './training/cp_mlp_save4.ckpt'
# checkpoint_filepath = './training/cp_mlp_rw_discharge.ckpt'
checkpoint_filepath = './training/cp_mlp_rw_discharge_batt_{}.ckpt'.format(BATTERY_NUM)

# base_checkpoint_filepath = './training/cp_mlp_save4.ckpt'
# base_model = get_model(batch_input_shape=(36,1,1), dt=dt, mlp=True, share_q_r=False)
# base_model.load_weights(base_checkpoint_filepath)
# base_weights = base_model.get_weights()

# change background scales
# q_max_base = 1.0e4
q_max_base = 1.0e5
R_0_base = 1.0e1

model_eval = get_model(batch_input_shape=(1,time_window_size-SIMULATION_OVER_STEPS,1), dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model_eval.compile(optimizer='adam', loss="mse", metrics=["mae"])
model = get_model(batch_input_shape=inputs.shape, dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model.compile(optimizer='adam', loss="mse", metrics=["mae"])

model.layers[0].cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))
MLPn_weigths = np.load('./training/MLPn_best_weights.npy',allow_pickle=True)
model.layers[0].cell.MLPn.set_weights([MLPn_weigths[:1], MLPn_weigths[1]])

xi = np.linspace(0.0,1.0,100)
fig = plt.figure('MLPp')
plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]), color='gray')

fig = plt.figure('MLPn')
plt.plot(xi, model.layers[0].cell.MLPn(xi[:,np.newaxis]), color='gray')


# model.load_weights(checkpoint_filepath)
model.set_weights(np.load('./training/model_weights_rw_disc_batt_{}.npy'.format(BATTERY_NUM), allow_pickle=True))
weights = model.get_weights()

if args.save:
    np.save('./training/model_weights_rw_disc_batt_{}.npy'.format(BATTERY_NUM), weights)

# weights = base_weights.copy()
# weights[0] = base_weights[0][1::3][:8]
# weights[1] = base_weights[1][1::3][:8]

# weights[0] = np.concatenate([np.tile(base_weights[0][1::3][i], 10) for i in range(8)])

# model.set_weights(weights)


pred_shiffed = model.predict(inputs_shiffed)[:,:,0]
# print('Model Eval [mse,mae]:', model_eval.evaluate(inputs_shiffed[:,:-SIMULATION_OVER_STEPS,:], target_shiffed))

# pred = model.predict(inputs)
pred = np.full((inputs.shape[0],inputs.shape[1]), np.nan)
for i in range(pred.shape[0]):
    pred[i, :(reach_EOD[i]+SIMULATION_OVER_STEPS)] = pred_shiffed[i, (max_size - reach_EOD[i]):]


# get mse dist
mse = np.zeros(inputs.shape[0])
weights_eval = weights.copy()
for i in range(inputs.shape[0]):
    weights_eval[0] = np.reshape(weights[0][i], (1,))
    weights_eval[1] = np.reshape(weights[1][i], (1,))
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
    fig.savefig('./figures/mse_dist_rw_disc_batt_{}.png'.format(BATTERY_NUM))


fig = plt.figure()
plt.hist(weights[0]*model.layers[0].cell.qMaxBASE.numpy())
plt.xlabel(r'$q_{max}$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

if args.save:
    fig.savefig('./figures/q_max_dist_rw_disc_batt_{}.png'.format(BATTERY_NUM))

fig = plt.figure('q_max_over_time')
plt.plot(inputs_time[:,0]/3600, weights[0]*model.layers[0].cell.qMaxBASE.numpy(), '.', label='RW. Dis.')
# plt.xlabel(r'Sample')
plt.xlabel('Time (h)')
plt.ylabel(r'$q_{max}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()

if args.save:
    fig.savefig('./figures/q_max_time_rw_disc_batt_{}.png'.format(BATTERY_NUM))

fig = plt.figure()
plt.hist(weights[1]*model.layers[0].cell.RoBASE.numpy())
plt.xlabel(r'$R_0$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

if args.save:
    fig.savefig('./figures/R_0_dist_rw_disc_batt_{}.png'.format(BATTERY_NUM))

fig = plt.figure('R_0_over_time')
plt.plot(inputs_time[:,0]/3600, weights[1]*model.layers[0].cell.RoBASE.numpy(), '.', label='RW. Dis.')
# plt.xlabel(r'Sample')
plt.xlabel('Time (h)')
plt.ylabel(r'$R_0$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()

if args.save:
    fig.savefig('./figures/R_0_time_rw_disc_batt_{}.png'.format(BATTERY_NUM))

data_to_save = {
    'time': inputs_time[:,0],
    'q_max': weights[0]*model.layers[0].cell.qMaxBASE.numpy(),
    'R_0': weights[1]*model.layers[0].cell.RoBASE.numpy()
}

if args.save:
    np.save('./training/q_max_R_0_rw_disc_{}.npy'.format(BATTERY_NUM),  data_to_save)

F = 96487.0
V_INT_k = lambda x,i: (2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i)
V_INT = lambda x,A: np.dot(A, np.array([V_INT_k(x,i) for i in range(len(A))])) / F
def Ai(A,i,a):
    A[i]=a
    return A
# reference Redlich-Kister coeff vals
Ap = np.array([-31593.7, 0.106747, 24606.4, -78561.9, 13317.9, 307387.0, 84916.1, -1.07469e+06, 2285.04, 990894.0, 283920, -161513, -469218])

xi = np.linspace(0.0,1.0,100)
Vint = np.array([V_INT(x,Ap) for x in xi])
fig = plt.figure('MLPp')
plt.plot(xi,Vint, label='Redlich V_INT')
plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]), label='MLP V_INT')
plt.grid()
plt.legend()

fig = plt.figure('MLPn')
plt.plot(xi, model.layers[0].cell.MLPn(xi[:,np.newaxis]))
plt.grid()


time_axis = np.arange(time_window_size) * dt
cmap = matplotlib.cm.get_cmap('Spectral')


sel_to_plot = np.arange(0,4,dtype=int)
fig = plt.figure()

# plt.subplot(311)
# for i in range(inputs.shape[0]):
#     plt.plot(time_axis, inputs[i,:,0])
# plt.ylabel('Current (A)')
# plt.grid()

plt.subplot(211)
# plt.plot(time_axis[0], target_shiffed[0,0], color='black', label='Actual')
# plt.plot(time_axis[0], pred_shiffed[0,0], '--', color='black', label='Predicted')
# # for i in range(3):
# for i in range(target_shiffed.shape[0]):
#     # plt.plot(time_axis[:-SIMULATION_OVER_STEPS], target_shiffed[i,:], color='gray')
#     plt.plot(time_axis, target_shiffed[i,:], color='C{}'.format(i), alpha=0.5)
# # for i in range(3):
# for i in range(pred_shiffed.shape[0]):
#     idx_end = len(time_axis)
#     idx = np.argwhere(pred_shiffed[i,:]<EOD)
#     if len(idx):
#         idx_end = idx[0][0]
#     plt.plot(time_axis[:idx_end], pred_shiffed[i,:idx_end], '--', color='C{}'.format(i))
# plt.ylabel('Voltage (V)')

for i in sel_to_plot:
# for i in range(inputs.shape[0]):
    # plt.plot(time_axis[:-SIMULATION_OVER_STEPS], target_shiffed[i,:], color='gray')
    plt.plot(time_axis, inputs[i,:,0], color='C{}'.format(i), alpha=0.5)
plt.ylabel('Current (A)')

plt.grid()
# plt.legend(loc="upper right", ncol=2, bbox_to_anchor=(0,1.02,1,0.2), borderaxespad=0)


plt.subplot(212)
for i in sel_to_plot:
# for i in range(target.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], target[i,:], color='C{}'.format(i), alpha=0.5)
    # plt.plot(time_axis, target[i,:], color='C{}'.format(i), alpha=0.5)
for i in sel_to_plot:
# for i in range(pred.shape[0]):
    # idx_end = len(time_axis)
    # idx = np.argwhere(pred[i,:]<EOD)
    idx_end = np.argmax((pred[i,:]<EOD) & (np.isnan(np.concatenate([target[i,:],np.ones(SIMULATION_OVER_STEPS, dtype=bool)]))))
    if idx_end<=0:
        idx_end = len(time_axis)
    plt.plot(time_axis[:idx_end], pred[i,:idx_end], '--', color='C{}'.format(i))
plt.ylabel('Voltage (V)')
# plt.ylim([3.0,4.2])
plt.grid()

plt.xlabel('Time (s)')

if args.save:
    fig.savefig('./figures/I_V_pred_rw_disc_batt_{}.png'.format(BATTERY_NUM))

# plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="upper right", mode="expand", borderaxespad=0, ncol=2)


fig = plt.figure()

plt.subplot(211)
for i in range(pred_shiffed.shape[0]):
    # plt.plot(time_axis[:-SIMULATION_OVER_STEPS], pred_shiffed[i,:-SIMULATION_OVER_STEPS] / target_shiffed[i,:])
    # plt.plot(pred_shiffed[i,:] / target_shiffed[i,:])
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], inputs[i,:-SIMULATION_OVER_STEPS,0])

# plt.ylabel('Pred / Actual Ratio (V)')
plt.ylabel('Current (A)')
plt.grid()

plt.subplot(212)
for i in range(pred.shape[0]):
    # plt.plot(time_axis[:-SIMULATION_OVER_STEPS], pred[i,:-SIMULATION_OVER_STEPS] / target[i,:])
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS],pred[i,:-SIMULATION_OVER_STEPS] / target[i,:])

plt.ylabel('Pred / Actual Ratio (V)')
# plt.ylim([3.0,4.2])
plt.grid()

plt.xlabel('Time (s)')

if args.save:
    fig.savefig('./figures/pred_ratio_rw_disc_batt_{}.png'.format(BATTERY_NUM))


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
    fig.savefig('./figures/EOD_pred_rw_disc_batt_{}.png'.format(BATTERY_NUM))

# %%
plt.show()
