import numpy as np
import math
from time import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from model import get_model

from battery_data import getDischargeMultipleBatteries

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)


class resetStateCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_state=None):
        self.initial_state=initial_state

    def on_epoch_begin(self, epoch, logs=None):
        if self.initial_state is None:
            self.initial_state = self.model.layers[0].cell.get_initial_state(batch_size=self.model.input.shape[0])
        self.model.layers[0].reset_states(self.initial_state)

class InputSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, time_window_size):
        self.x, self.y = x_set, y_set
        self.batch_size = self.x.shape[0]
        self.time_window_size = time_window_size
        self.time_size = self.x.shape[1]

    def __len__(self):
        return math.ceil(self.time_size / self.time_window_size)

    def __getitem__(self, idx):
        batch_x = self.x[:, idx * self.time_window_size:(idx + 1) * self.time_window_size, :]
        batch_y = self.y[:, idx * self.time_window_size:(idx + 1) * self.time_window_size, :]

        return batch_x, batch_y


# load battery data
data_RW = getDischargeMultipleBatteries()
max_idx_to_use = 3
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


# # generate artificial data
# BATCH_SIZE = 36
# dt = 10.0
# time_window_size = 700
# inputs = np.ones((time_window_size,BATCH_SIZE), dtype=DTYPE) * np.linspace(1.0,2.0,BATCH_SIZE)  # uniform constant load
# inputs = inputs.T[:,:,np.newaxis]
# model = get_model(batch_input_shape=inputs.shape, dt=dt)
# model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
# target = model.predict(inputs)


# move timesteps with earlier EOD
EOD = 3.2
V_0 = 4.19135029  # V adding when I=0 for the shift
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


time_window_size = inputs.shape[1]  # 310

checkpoint_filepath = './training/cp_mlp_save4.ckpt'

SIMULATION_OVER_STEPS = 200
inputs_shiffed = np.hstack([inputs_shiffed, inputs_shiffed[:, -SIMULATION_OVER_STEPS:]])
inputs = np.hstack([inputs, inputs[:, -SIMULATION_OVER_STEPS:]])
time_window_size = inputs_shiffed.shape[1]

val_idx = np.linspace(0,35,6,dtype=int)
train_idx = [i for i in np.arange(0,36) if i not in val_idx]

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

model_eval = get_model(batch_input_shape=(1,time_window_size-SIMULATION_OVER_STEPS,1), dt=dt, mlp=True, share_q_r=False)
model_eval.compile(optimizer='adam', loss="mse", metrics=["mae"])
model = get_model(batch_input_shape=inputs.shape, dt=dt, mlp=True, share_q_r=False)
model.compile(optimizer='adam', loss="mse", metrics=["mae"])


xi = np.linspace(0.0,1.0,100)
fig = plt.figure('MLPp')
plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]), color='gray')

fig = plt.figure('MLPn')
plt.plot(xi, model.layers[0].cell.MLPn(xi[:,np.newaxis]), color='gray')


model.load_weights(checkpoint_filepath)
# avg_mlp_weights = np.load('mlp_all_avg_weights.npy',allow_pickle=True)
# avg_mlp_weights[0] = np.array([avg_mlp_weights[0]])
# avg_mlp_weights[1] = np.array([avg_mlp_weights[1]])
# model.set_weights(avg_mlp_weights)

weights = model.get_weights()
print(weights)
# np.save('mlp_all_avg_weights.npy', model.get_weights())



pred_shiffed = model.predict(inputs_shiffed)[:,:,0]
# print('Model Eval [mse,mae]:', model_eval.evaluate(inputs_shiffed[:,:-SIMULATION_OVER_STEPS,:], target_shiffed))

# pred = model.predict(inputs)
pred = np.full((inputs.shape[0],inputs.shape[1]), np.nan)
for i in range(pred.shape[0]):
    pred[i, :(reach_EOD[i]+SIMULATION_OVER_STEPS)] = pred_shiffed[i, (max_size - reach_EOD[i]):]


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

fig = plt.figure()
plt.hist(weights[0]*model.layers[0].cell.qMaxBASE.numpy())
plt.xlabel(r'$q_{max}$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

fig = plt.figure()
plt.hist(weights[1]*model.layers[0].cell.RoBASE.numpy())
plt.xlabel(r'$R_0$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))


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
# plt.ylim([3.0,4.2])
plt.grid()

plt.xlabel('Time (s)')


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


xi = np.linspace(0.0,1.0,100)
fig = plt.figure('MLPp')
plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]))
plt.grid()

fig = plt.figure('MLPn')
plt.plot(xi, model.layers[0].cell.MLPn(xi[:,np.newaxis]))
plt.grid()

# np.save('MLPp_best_weights.npy', model.layers[0].cell.MLPp.get_weights())
# np.save('MLPn_best_weights.npy', model.layers[0].cell.MLPn.get_weights())

fig, ax1 = plt.subplots()
x_axis = np.linspace(0.0,1.0,len(time_axis[:-SIMULATION_OVER_STEPS]))
for i in range(target.shape[0]):
    ax1.plot(x_axis, target[i,:], color='gray')

mlp_pred = model.layers[0].cell.MLPp(xi[:,np.newaxis])
Y = np.hstack([np.linspace(0.85,-0.2,90), np.linspace(-0.25,-0.8,10)])
ax2 = ax1.twinx()
ax2.plot(xi, Y)
ax2.set_ylim([-1.0,1.0])
plt.grid()


pred_lb = []
pred_ub = []
for i in range(pred.shape[1]):
    up = np.percentile(pred[:,i], 92.5)
    lb = np.percentile(pred[:,i], 7.5)
    if up<EOD:
        break
    pred_ub += [up]
    pred_lb += [lb]

total_samples_pts = np.sum(~np.isnan(target_all[val_idx,:].ravel()))

fig = plt.figure()
plt.fill_between(range(len(pred_ub)), pred_ub, pred_lb, facecolor='blue', alpha=0.3, label='85% CI')
plt.plot(target_all[val_idx[0],0], label='Test Samples', color='black')
plt.plot(target_all[val_idx,:].T)

within_CI = 0
for i in range(target_all.shape[1]):
    within = (target_all[val_idx,i]<=pred_ub[i]) & (target_all[val_idx,i]>=pred_lb[i])
    within_CI += np.sum(within)
    plt.plot(i*np.ones(np.sum(~within)), target_all[val_idx,i][~within], '+k', markersize=3)
plt.plot(i*np.ones(np.sum(~within)), target_all[val_idx,i][~within], '+k', markersize=3, label='Pts out CI - {:.1f}%'.format((total_samples_pts - within_CI) / total_samples_pts*100))

plt.ylim([3.2,4.2])
plt.grid()
plt.legend()

plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')

# model_eval = get_model(batch_input_shape=(1,time_window_size,1), dt=dt, mlp=True, share_q_r=False)
# model_eval.compile(optimizer='adam', loss="mse", metrics=["mae"])

# q_max_lb = np.percentile(weights[0], 2.5)
# q_max_ub = np.percentile(weights[0], 97.5)
# q_max_ub = np.percentile(weights[0], 50.0)

# R_0_lb = np.percentile(weights[1], 2.5)
# R_0_up = np.percentile(weights[1], 97.5)
# R_0_mean = np.percentile(weights[1], 50.0)

# weights_eval[0] = np.reshape(q_max_lb, (1,))
# weights_eval[1] = np.reshape(R_0_up, (1,))
# model_eval.set_weights(weights_eval)
# # pred_lb_shiffed = np.mean(model_eval.predict(inputs_shiffed), axis=0)
# # pred_lb = pred_lb_shiffed[:, (max_size - np.mean(reach_EOD)):]
# pred_lb = np.mean(model_eval.predict(inputs), axis=0)

# weights_eval[0] = np.reshape(q_max_ub, (1,))
# weights_eval[1] = np.reshape(R_0_lb, (1,))
# model_eval.set_weights(weights_eval)
# # pred_ub_shiffed = np.mean(model_eval.predict(inputs_shiffed), axis=0)
# # pred_ub = pred_ub_shiffed[:, (max_size - np.mean(reach_EOD)):]
# pred_ub = np.mean(model_eval.predict(inputs), axis=0)

# fig = plt.figure()
# plt.plot(pred_lb, color='gray',linewidth=3.0)
# plt.plot(pred_ub, color='gray',linewidth=3.0)
# # plt.fill_between(time_axis, pred_lb, pred_ub)
# plt.plot(target_all[val_idx,:].T)

plt.show()
