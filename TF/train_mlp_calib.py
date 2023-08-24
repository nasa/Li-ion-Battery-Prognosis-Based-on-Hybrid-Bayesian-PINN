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
        # if self.initial_state is None:
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
model = get_model(batch_input_shape=(1,time_window_size,inputs.shape[2]), dt=dt, mlp=True, stateful=True, mlp_trainable=False)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse", metrics=["mae"])
model.summary()

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

checkpoint_filepath = './training/cp_mlp_cal.ckpt'
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
    min_delta=1e-6,
    verbose=1,
    mode='min')


EPOCHS = 50

# callbacks = [model_checkpoint_callback,reduce_lr_on_plateau, resetStateCallback()]
# callbacks = [model_checkpoint_callback,reduce_lr_on_plateau]
callbacks = [resetStateCallback()]

# tf.debugging.enable_check_numerics()

# val_idx = np.linspace(0,35,6,dtype=int)
# train_idx = [i for i in np.arange(0,36) if i not in val_idx]

#load pre-trained weights
avg_mlp_weights = np.load('mlp_all_avg_weights.npy',allow_pickle=True)
# model.set_weights(avg_mlp_weights)

# calibrate models individualy
# for i in range(inputs_shiffed.shape[0]):
#     print("")
#     print("* * Training Batch {}/{} * *".format(i,inputs_shiffed.shape[0]))

#     model.set_weights(avg_mlp_weights)
#     history = model.fit(inputs_shiffed[i,:,:][np.newaxis,:,:], target_shiffed[i,:][np.newaxis,:], epochs=EPOCHS, callbacks=callbacks)
#     np.save('./training/mlp_weights_batch-{}.npy'.format(i), model.get_weights())

# calibrate models together (shared MLP indiv. q_max and R_0)
history = model.fit(inputs_shiffed, target_shiffed, epochs=EPOCHS, callbacks=callbacks)

np.save('./training/history_mlp_cal.npy', history.history)
