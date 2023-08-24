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


val_idx = np.linspace(0,35,6,dtype=int)
train_idx = [i for i in np.arange(0,36) if i not in val_idx]

time_window_size = inputs.shape[1]  # 310
model = get_model(batch_input_shape=(inputs[train_idx,:,:].shape[0],time_window_size,inputs.shape[2]), dt=dt, mlp=True, share_q_r=False, stateful=True)
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse", metrics=["mae"], sample_weight_mode="temporal")
model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss="mse", metrics=["mae"])
model.summary()

# samples_intervals           = [100, 100, 150, 100, 200,  50,  40, max_size-740]
# samples_intervals_weights   = [1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 10.0         ]

# sample_weight = np.hstack([
#     np.ones((BATCH_SIZE, samples_intervals[i]))*samples_intervals_weights[i]
#     for i in range(len(samples_intervals))
# ])

# # add noise to original weights
# gt_weights = model.get_weights()
# model.set_weights(np.array(gt_weights) + np.random.normal(0, .1, len(gt_weights)))

# print("Ground Truth weights:", gt_weights)
# print("Noisy weights:", model.get_weights())

# print(model.layers[0].cell.getAparams())

# # target = train_data[:,:,np.newaxis]
# data_gen = InputSequence(inputs,target,time_window_size)

# inputs = inputs_shiffed
# target = target_shiffed

checkpoint_filepath = './training/cp_mlp.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    verbose=1,
    save_best_only=True)
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.5, 
    min_lr=1e-6, 
    patience=25,
    min_delta=1e-10,
    verbose=1,
    mode='min')


def scheduler(epoch):
    if epoch < 200:
        return 2e-2
    elif epoch < 300:
        return 1e-2
    elif epoch < 1500:
        return 5e-3
    elif epoch < 5000:
        return 2e-3
    else:
        return 1e-3

scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

EPOCHS = 3000

# callbacks = [model_checkpoint_callback,scheduler_cb, reduce_lr_on_plateau]
callbacks = [model_checkpoint_callback,reduce_lr_on_plateau]
# callbacks = []

#load pre-trained weights
# model.set_weights(np.load('./training/mlp_all_avg_weights.npy',allow_pickle=True))

start = time()
# history = model.fit(inputs_shiffed, target_shiffed[:,:,np.newaxis], epochs=EPOCHS, callbacks=callbacks, shuffle=False, sample_weight=sample_weight)
history = model.fit(inputs_shiffed[train_idx,:,:], target_shiffed[train_idx,:,np.newaxis], epochs=EPOCHS, callbacks=callbacks, shuffle=False)
duration = time() - start
print("Train time: {:.2f} s - {:.3f} s/epoch ".format(duration, duration/EPOCHS))

np.save('./training/history_mlp.npy', history.history)
