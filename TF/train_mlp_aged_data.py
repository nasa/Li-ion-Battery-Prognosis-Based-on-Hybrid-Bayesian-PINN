import numpy as np
import math
from time import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from model import get_model

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)

# load all battery data
data_refer = np.load('./training/input_data_refer_disc_batt_1to8.npy', allow_pickle=True).item()

inputs = data_refer['inputs']
target = data_refer['target']
inputs_time = data_refer['time']

time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]
dt = inputs_time[0,3]-inputs_time[0,2]


# move timesteps with earlier EOD
EOD = 3.2

inputs_shiffed = inputs.copy()
target_shiffed = target.copy()
sample_weight = np.zeros((BATCH_SIZE, time_window_size))
reach_EOD = np.ones(BATCH_SIZE, dtype=int) * time_window_size
for row in np.argwhere((target<EOD) | (np.isnan(target))):
    if reach_EOD[row[0]]>row[1]:
        reach_EOD[row[0]]=row[1]
        inputs_shiffed[row[0],:,0] = np.zeros(time_window_size)
        inputs_shiffed[row[0],:,0][time_window_size-row[1]:] = inputs[row[0],:,0][:row[1]]
        sample_weight[row[0],:][time_window_size-row[1]:] = 1.0
        target_shiffed[row[0]] = np.ones(time_window_size) * target[row[0]][0]
        target_shiffed[row[0]][time_window_size-row[1]:] = target[row[0]][:row[1]]


q_max_base = 1.0e3
R_0_base = 1.0e1

model = get_model(batch_input_shape=(BATCH_SIZE,time_window_size,inputs.shape[2]), dt=dt, mlp=True, mlp_trainable=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss="mse", metrics=["mae"], sample_weight_mode="temporal")
model.summary()


# weight_filepath = './training/cp_mlp_save4.ckpt'
# base_model = get_model(batch_input_shape=(36,1,1), dt=dt, mlp=True, share_q_r=False)
# base_model.load_weights(weight_filepath)
# base_weights = base_model.get_weights()

# weights = base_weights.copy()
# weights[0] = base_weights[0][2] * np.ones(BATCH_SIZE)
# weights[1] = base_weights[1][2] * np.ones(BATCH_SIZE)
# model.set_weights(weights)

# set saved mlp weights
model.layers[0].cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))
# MLPn_weigths = np.load('./training/MLPn_best_weights.npy',allow_pickle=True)
# model.layers[0].cell.MLPn.set_weights([MLPn_weigths[:1], MLPn_weigths[1]])


# checkpoint_filepath = './training/cp_mlp_aged_batt_{}.ckpt'.format(BATTERY_NUM)
checkpoint_filepath = './training/cp_mlp_aged_batt_1to8.ckpt'
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
    patience=20,
    min_delta=1e-9,
    verbose=1,
    mode='min')


EPOCHS = 3000

# callbacks = [model_checkpoint_callback,reduce_lr_on_plateau,resetStateCallback()]
# callbacks = [model_checkpoint_callback,scheduler_cb, resetStateCallback()]
callbacks = [model_checkpoint_callback,reduce_lr_on_plateau]
# callbacks = []

history = model.fit(inputs_shiffed, target_shiffed[:,:,np.newaxis], epochs=EPOCHS, callbacks=callbacks, shuffle=False, sample_weight=sample_weight)

# np.save('./training/history_mlp_aged_batt_{}.npy'.format(BATTERY_NUM), history.history)
np.save('./training/history_mlp_aged_batt_1to8.npy', history.history)
