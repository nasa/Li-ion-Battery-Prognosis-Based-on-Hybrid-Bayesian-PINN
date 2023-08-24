# %% imports
import numpy as np
import math
from time import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from model import get_model

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)

# %% load all battery data
data_refer = np.load('./training/input_data_refer_disc_batt_1to8.npy', allow_pickle=True).item()
data_rw = np.load('./training/input_data_rw_disc_batt_1to8.npy', allow_pickle=True).item()

inputs_refer = data_refer['inputs']
target_refer = data_refer['target']
inputs_time_refer = data_refer['time']

# subsample - every 10 points
inputs_rw = data_rw['inputs'][:,::10,:]
target_rw = data_rw['target'][:,::10]
inputs_time_rw = data_rw['time'][:,::10]

# %% merge and mix refer and rw
rw_ext_size = inputs_refer.shape[1] - inputs_rw.shape[1]
rw_ext = np.full((inputs_rw.shape[0],rw_ext_size), np.nan)

inputs_rw_ext = np.hstack([inputs_rw[:,:,0],rw_ext])[:,:,np.newaxis]
target_rw_ext = np.hstack([target_rw,rw_ext])
inputs_time_rw_ext = np.hstack([inputs_time_rw,rw_ext])

inputs = np.vstack([inputs_refer, inputs_rw_ext])
target = np.vstack([target_refer, target_rw_ext])
inputs_time = np.vstack([inputs_time_refer, inputs_time_rw_ext])
# %%

time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]
dt = inputs_time[0,3]-inputs_time[0,2]
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
q_max_base = 1.0e3
R_0_base = 1.0e1


mse_loss = tf.keras.losses.MeanSquaredError()

def custom_loss(y_true, y_pred):
    return mse_loss(y_true, y_pred) + mse_loss(y_true[:,-1], y_pred[:,-1])

model = get_model(batch_input_shape=(BATCH_SIZE,time_window_size,inputs.shape[2]), dt=dt, mlp=True, mlp_trainable=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-3), loss="mse", metrics=["mae"])

model.summary()

# set saved mlp weights
cell = model.layers[0].cell
# cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))
# MLPn_weigths = np.load('./training/MLPn_best_weights.npy',allow_pickle=True)
# cell.MLPn.set_weights([MLPn_weigths[:1], MLPn_weigths[1]])


checkpoint_filepath = './training/cp_mlp_MIX_discharge_batt_1to8.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    verbose=1,
    save_best_only=True)
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.75, 
    min_lr=1e-6, 
    patience=20,
    min_delta=1e-7,
    verbose=1,
    mode='min')


def scheduler(epoch):
    if epoch < 50:
        return 5e-2
    if epoch < 100:
        return 2e-2
    elif epoch < 200:
        return 1e-2
    elif epoch < 500:
        return 5e-3
    elif epoch < 1000:
        return 2e-3
    else:
        return 1e-3

scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

EPOCHS = 5000

# callbacks = [model_checkpoint_callback,reduce_lr_on_plateau,scheduler_cb]
callbacks = [model_checkpoint_callback,reduce_lr_on_plateau]
# callbacks = []

history = model.fit(inputs_shiffed, target_shiffed[:,:,np.newaxis], epochs=EPOCHS, callbacks=callbacks, shuffle=False)

np.save('./training/history_mlp_MIX_discharge_batt_1to8.npy', history.history)
