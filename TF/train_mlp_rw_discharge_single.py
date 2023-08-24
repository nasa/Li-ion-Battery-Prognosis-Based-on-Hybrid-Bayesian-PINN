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

# %% load all battery data
data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type='discharge (random walk)')

# %% process data
num_seq = 1  # number of sequences of rw discharge of each battery to include
num_batteries = 1  # up until battery to include (9 to 12 has diff discharge profile)

max_size = 0
inputs = []
inputs_time = []
target = []

for k,rw_data in data_RW.items():
    if k>num_batteries:
        continue
    
    time = np.hstack([rw_data[2][i] for i in range(len(rw_data[2]))])
    time = time - time[0]
    current_inputs = np.hstack([rw_data[1][i] for i in range(len(rw_data[1]))])
    voltage_target = np.hstack([rw_data[0][i] for i in range(len(rw_data[0]))])

    last_idx = 0
    seq_durations = np.diff([0]+list(np.argwhere(np.diff(time)>10)[:,0]+1))
    
    for curr_duration in seq_durations[:num_seq]:
        if curr_duration>max_size:
            max_size = curr_duration
        curr_idx = last_idx + curr_duration
        inputs.append(current_inputs[last_idx:curr_idx])
        inputs_time.append(time[last_idx:curr_idx])
        target.append(voltage_target[last_idx:curr_idx])
        last_idx = curr_idx

# add nan to end of seq to have all seq in same size
for i in range(len(inputs)):
    prep_inputs = np.full(max_size, np.nan)
    prep_target = np.full(max_size, np.nan)
    prep_inputs_time = np.full(max_size, np.nan)
    prep_inputs[:len(inputs[i])] = inputs[i]
    prep_target[:len(target[i])] = target[i]
    prep_inputs_time[:len(inputs_time[i])] = inputs_time[i]
    inputs[i] = prep_inputs
    target[i] = prep_target
    inputs_time[i] = prep_inputs_time

inputs = np.vstack(inputs)[:,:,np.newaxis]
target = np.vstack(target)
inputs_time = np.vstack(inputs_time)

time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]
dt = np.diff(data_RW[1][2,0])[1]
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


# change background scales
q_max_base = 1.0e4
R_0_base = 1.0e1

model = get_model(batch_input_shape=(BATCH_SIZE,time_window_size,inputs.shape[2]), dt=dt, mlp=True, mlp_trainable=False, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-2), loss="mse", metrics=["mae"])

model.summary()


weight_filepath = './training/cp_mlp_save4.ckpt'
base_model = get_model(batch_input_shape=(36,1,1), dt=dt, mlp=True, share_q_r=False)
base_model.load_weights(weight_filepath)
base_weights = base_model.get_weights()

weights = model.get_weights()

# base model saved background scales
q_max_base_saved = 1.0e4
R_0_base_saved = 1.0e1

# # get second sample of every battery as initial weight (q_max and R_0)
# # weights[0] = np.concatenate([np.tile(base_weights[0][1::3][i]*(q_max_base_saved/q_max_base), num_seq) for i in range(num_batteries)])
# # weights[1] = np.concatenate([np.tile(base_weights[1][1::3][i]*(R_0_base_saved/R_0_base), num_seq) for i in range(num_batteries)])
# model.set_weights(weights)

# set saved mlp weights
cell = model.layers[0].cell
cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))
MLPn_weigths = np.load('./training/MLPn_best_weights.npy',allow_pickle=True)
cell.MLPn.set_weights([MLPn_weigths[:1], MLPn_weigths[1]])


checkpoint_filepath = './training/cp_mlp_rw_discharge_single.ckpt'
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
    patience=10,
    min_delta=1e-7,
    verbose=1,
    mode='min')


def scheduler(epoch):
    if epoch < 200:
        return 2e-2
    elif epoch < 300:
        return 1e-2
    elif epoch < 1000:
        return 5e-3
    elif epoch < 2000:
        return 2e-3
    else:
        return 1e-3

scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

EPOCHS = 3000

# callbacks = [model_checkpoint_callback,reduce_lr_on_plateau,scheduler_cb]
callbacks = [model_checkpoint_callback,reduce_lr_on_plateau]
# callbacks = []

history = model.fit(inputs_shiffed, target_shiffed[:,:,np.newaxis], epochs=EPOCHS, callbacks=callbacks, shuffle=False)

np.save('./training/history_mlp_rw_discharge_single.npy', history.history)
