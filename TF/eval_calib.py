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


model_eval = get_model(batch_input_shape=(1,time_window_size,inputs.shape[2]), dt=dt, mlp=True)
model_eval.compile(optimizer='adam', loss="mse", metrics=["mae"])

inputs_shiffed = np.hstack([inputs_shiffed, inputs_shiffed[:, -200:]])
inputs = np.hstack([inputs, inputs[:, -400:-200]])
time_window_size = inputs_shiffed.shape[1]

model = get_model(batch_input_shape=(1,time_window_size,inputs.shape[2]), dt=dt, mlp=True)
model.compile(optimizer='adam', loss="mse", metrics=["mae"])

pred = np.full((inputs.shape[0],inputs.shape[1]), np.nan)
pred_shiffed = np.zeros((inputs_shiffed.shape[0],inputs_shiffed.shape[1]))
q_max_r_0_weights = np.zeros((target.shape[0],2))

mse = np.zeros(target.shape[0])

for i in range(inputs_shiffed.shape[0]):
    print("")
    print("* * Predict Batch {}/{} * *".format(i,inputs_shiffed.shape[0]))
    weights = np.load('./training/mlp_weights_batch-{}.npy'.format(i),allow_pickle=True)
    q_max_r_0_weights[i,:] = np.array(weights[0:2], dtype='float64')
    weights[0] = np.array([weights[0]])
    weights[1] = np.array([weights[1]])
    model.set_weights(weights)
    model_eval.set_weights(weights)
    pred_shiffed[i, :] = model.predict(inputs_shiffed[i,:,:][np.newaxis,:,:])[0,:,0]
    # pred[i, :] = model.predict(inputs[i,:,:][np.newaxis,:,:])[0,:,0]
    pred[i, :(reach_EOD[i]+200)] = pred_shiffed[i, (max_size - reach_EOD[i]):]
    mse[i] = model_eval.evaluate(inputs_shiffed[i,:target_shiffed.shape[1],:][np.newaxis,:,:], target_shiffed[i,:][np.newaxis,:,np.newaxis])[0]
    print("MSE:, ", mse[i])

print("")
print("AVG MSE:, ", mse.mean())

print('pred_shiffed NAN sum:', np.sum(np.isnan(pred_shiffed)))
print('pred NAN sum:', np.sum(np.isnan(pred)))

np.save('./training/q_max_r_0_weights_dist.npy', q_max_r_0_weights)

time_axis = np.arange(time_window_size) * dt
cmap = matplotlib.cm.get_cmap('Spectral')

fig = plt.figure()
plt.hist(mse)
plt.xlabel(r'mse')

fig = plt.figure()
plt.hist(q_max_r_0_weights[:,0])
plt.xlabel(r'$q_{max}$')

fig = plt.figure()
plt.hist(q_max_r_0_weights[:,1])
plt.xlabel(r'$R_0$')

fig = plt.figure()

plt.subplot(211)
for i in range(pred_shiffed.shape[0]):
    plt.plot(time_axis[:target_shiffed.shape[1]], target_shiffed[i,:], color='gray')
for i in range(pred_shiffed.shape[0]):
    plt.plot(time_axis, pred_shiffed[i,:])
plt.ylabel('Voltage (V)')
plt.grid()

plt.subplot(212)
for i in range(pred.shape[0]):
    plt.plot(time_axis[:target.shape[1]], target[i,:], color='gray')
for i in range(pred.shape[0]):
    plt.plot(time_axis, pred[i,:])
plt.ylabel('Voltage (V)')
plt.ylim([3.0,4.2])
plt.grid()

plt.xlabel('Time (s)')


reach_EOD_pred = np.ones(BATCH_SIZE, dtype=int) * time_window_size
for row in np.argwhere(pred<EOD):
    if reach_EOD_pred[row[0]]>row[1]:
        reach_EOD_pred[row[0]]=row[1]

fig = plt.figure()
EOD_range = [np.min(reach_EOD*dt),np.max(reach_EOD*dt)]
plt.plot(EOD_range, EOD_range, '--k')
plt.plot(reach_EOD*dt, reach_EOD_pred*dt, '.')
plt.ylabel("Predicted EOD (s)")
plt.xlabel("Actual EOD (s)")
plt.xlim(EOD_range)
plt.ylim(EOD_range)
plt.grid()

plt.show()
