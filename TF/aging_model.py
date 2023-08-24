# %% imports
import numpy as np
import math
from time import time
import argparse
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

matplotlib.rc('font', size=14)

# %% load data
data = np.load('./training/input_data_refer_disc_batt_1to8.npy', allow_pickle=True).item()
inputs = data['inputs']
target = data['target']
inputs_time = data['time']

sizes = data['sizes']
init_time = data['init_time']

power_data = np.load('./training/input_data_power-hour_batt_1to8.npy', allow_pickle=True).item()

power_time = power_data['power_time']
time_all = power_data['time']
cycles = power_data['cycles']

data_q_max_R_0 = np.load('./training/q_max_R_0_aged_batt_1to8.npy', allow_pickle=True).item()
q_max_all = data_q_max_R_0['q_max']
R_0_all = data_q_max_R_0['R_0']

batt_index = []
for i,s in enumerate(sizes):
    batt_index += (list(np.ones(len(s), dtype=int)*i))

PWh = [(np.cumsum(power_time[i])/3.6e6) for i in range(len(power_time))]

# %% clean batt and val set
# batt_skip = [5]
# q_max_all = q_max
# q_max_idx = np.argwhere(np.array(batt_index) != 5)[:,0]
# q_max = q_max_all[q_max_idx]

# batt_index_all = np.array(batt_index, dtype=int)
# batt_index = batt_index_all[q_max_idx]

# %% capacity and q_max over time
fig, ax1 = plt.subplots()

for i in range(len(sizes)):
    # if i in batt_skip:
    #     continue
    ax1.plot(np.array(init_time[i])/3600,np.array(sizes[i])/360, 'o', fillstyle='none', color='C{}'.format(i))
ax1.set_ylabel('Capacity (Ah)')

ax1.set_xlabel('Time (h)')
ax1.grid(None)

ax2 = ax1.twinx()
for i in range(len(q_max_all)):
    ax2.plot(inputs_time[i,0]/3600, q_max_all[i], '.', color='C{}'.format(batt_index[i]))
ax2.set_ylabel(r'$q_{MAX}$')
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax1.plot([], [], 'o', fillstyle='none', color='black', label='Capacity')
ax1.plot([], [], '.', color='black', label=r'$q_{MAX}$')
ax1.legend(scatterpoints=1, loc='lower left')

for i in range(len(sizes)):
    # if i in batt_skip:
    #     continue
    ax2.plot([], [], '.', color='C{}'.format(i), label='Batt #{}'.format(i+1))
ax2.legend(loc='upper right')


# %%

fig, ax1 = plt.subplots()

for i in range(len(sizes)):
    idx = np.array([np.argwhere(time_all[i]>=init_time[i][j])[0][0] for j in range(len(init_time[i])) if len(np.argwhere(time_all[i]>=init_time[i][j]))])
    ax1.plot(PWh[i][idx],np.array(sizes[i])/360, 'o', fillstyle='none', color='C{}'.format(i))
ax1.set_ylabel('Capacity (Ah)')

ax1.set_xlabel('Cumulative Energy (kWh)')
ax1.grid(None)

ax2 = ax1.twinx()
X_all = np.zeros(q_max_all.shape)
for i in range(len(q_max_all)):
    idx = np.argwhere(time_all[batt_index[i]]>=inputs_time[i,0])[0][0]
    X_all[i] = PWh[batt_index[i]][idx]
    ax2.plot(X_all[i], q_max_all[i], '.', color='C{}'.format(batt_index[i]))
ax2.set_ylabel(r'$q_{MAX}$')
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax1.plot([], [], 'o', fillstyle='none', color='black', label='Capacity')
ax1.plot([], [], '.', color='black', label=r'$q_{MAX}$')
ax1.legend(scatterpoints=1, loc='lower left')

for i in range(len(sizes)):
    ax2.plot([], [], '.', color='C{}'.format(i), label='Batt #{}'.format(i+1))
ax2.legend(loc='upper right')


# %% R_0
fig, ax1 = plt.subplots()

# for i in range(len(sizes)):
#     # if i in batt_skip:
#     #     continue
#     idx = np.array([np.argwhere(time_all[i]>=init_time[i][j])[0][0] for j in range(len(init_time[i])) if len(np.argwhere(time_all[i]>=init_time[i][j]))])
#     ax1.plot(PWh[i][idx],np.array(sizes[i])/360, 'o', fillstyle='none', color='C{}'.format(i))
# ax1.set_ylabel('Capacity (Ah)')

# ax1.set_xlabel('Cumulative Energy (kWh)')
# ax1.grid(None)

# ax2 = ax1.twinx()
ax2 = ax1
ax2.grid(None)
for i in range(len(R_0_all)):
    idx = np.argwhere(time_all[batt_index[i]]>=inputs_time[i,0])[0][0]
    ax2.plot(X_all[i], R_0_all[i], '.', color='C{}'.format(batt_index[i]))
ax2.set_ylabel(r'$R_0$')
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax1.legend(scatterpoints=1, loc='lower left')

for i in range(len(sizes)):
    ax2.plot([], [], '.', color='C{}'.format(i), label='Batt #{}'.format(i+1))
ax2.legend(loc='upper right')


# %%
# model input data
q_max_idx = (np.array(batt_index) != 2) & (np.array(batt_index) != 5)
# q_max_idx = np.array(batt_index) != 3
X = X_all[q_max_idx]
q_max = q_max_all[q_max_idx]
R_0 = R_0_all[q_max_idx]

idx_sort = np.argsort(X)
max_q_max = np.max(q_max)
Y_q_max = q_max / max_q_max

max_X = np.max(X)
X_norm = X / max_X

max_R_0 = np.max(R_0)
Y_R_0 = R_0 / max_R_0

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

# %%
# class MeanStd(tf.keras.layers.Layer):
#     def __init__(self):
#         super(MeanStd, self).__init__()
#         self.mean = tf.keras.Sequential([
#             tf.keras.layers.Dense(6, activation='tanh'),
#             tf.keras.layers.Dense(2, activation='tanh'),
#             tf.keras.layers.Dense(1)
#         ])
#         self.std = tf.keras.Sequential([
#             # tf.keras.layers.Dense(2, activation='sigmoid'),
#             tf.keras.layers.Dense(1)
#         ])
    
#     def build(self, input_shape):
#         super(MeanStd, self).__init__()
#         self.mean.build(input_shape=input_shape)
#         self.std.build(input_shape=input_shape)

#     def call(self, inputs):
#         return tf.keras.layers.concatenate([self.mean(inputs), self.std(inputs)])

# model = tf.keras.Sequential([
#     MeanStd(),
#     tfp.layers.DistributionLambda(
#       lambda t: tfd.Normal(loc=t[..., :1],
#                            scale=1e-9 + tf.math.softplus(1.0 * t[...,1:]))),
# ])

# def scheduler(epoch):
#     if epoch < 100:
#         return 2e-2
#     elif epoch < 200:
#         return 1e-2
#     elif epoch < 500:
#         return 5e-3
#     elif epoch < 1000:
#         return 1e-3
#     else:
#         return 1e-4

# callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)]

# # Do inference.
# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=negloglik)
# history = model.fit(X_norm[idx_sort], Y[idx_sort], epochs=1500, verbose=1, callbacks=callbacks)

# # %%
# yhat = model(X_norm[idx_sort,np.newaxis])

# fig = plt.figure()
# plt.plot(X,q_max, '.', color='gray')
# plt.plot(X[idx_sort], yhat.quantile(0.025).numpy()[:,0] * max_q_max, 'b--', label='95% CI')
# plt.plot(X[idx_sort], yhat.quantile(0.5).numpy()[:,0] * max_q_max, 'b-', label='Median')
# plt.plot(X[idx_sort], yhat.quantile(0.975).numpy()[:,0] * max_q_max, 'b--')
# plt.grid()
# plt.xlabel('Cumulative Energy (kWh)')
# plt.ylabel(r'$q_{MAX}$')
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.legend()

# %%
# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    # c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                        scale=1e-5 + tf.math.softplus(t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.math.softplus(1.0 * t[...,n:])),
            reinterpreted_batch_ndims=1)),
    ])

# %%
class MeanStdVar(tf.keras.layers.Layer):
    def __init__(self):
        super(MeanStdVar, self).__init__()
        self.mean = tf.keras.Sequential([
            tfp.layers.DenseVariational(6, posterior_mean_field, prior_trainable, kl_weight=1/X.shape[0], activation='tanh'),
            tfp.layers.DenseVariational(2, posterior_mean_field, prior_trainable, kl_weight=1/X.shape[0], activation='tanh'),
            tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1/X.shape[0])
        ])
        self.std = tf.keras.Sequential([
            # tf.keras.layers.Dense(2, activation='sigmoid'),
            tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1/X.shape[0])
        ])
    
    def build(self, input_shape):
        super(MeanStdVar, self).__init__()
        # self.mean.build(input_shape=input_shape)
        # self.std.build(input_shape=input_shape)

    def call(self, inputs):
        return tf.keras.layers.concatenate([self.mean(inputs), self.std(inputs)])

    def get_dist_weights(self, dist='_posterior'):
        w=[]
        for l in self.mean.layers:
            w.append(getattr(l, dist).get_weights())
        for l in self.std.layers:
            w.append(getattr(l, dist).get_weights())
        return w

    def set_dist_weights(self, w, dist='_posterior'):
        for i,l in enumerate(self.mean.layers):
            getattr(l, dist).set_weights(w[i])
        for j,l in enumerate(self.std.layers):
            getattr(l, dist).set_weights(w[i+1+j])

    def set_dist_trainable(self, trainable=True, dist='_posterior'):
        for l in self.mean.layers:
            getattr(l, dist).layers[0].trainable = trainable
            # pass
        for l in self.std.layers:
            getattr(l, dist).layers[0].trainable = trainable

# %% q_max model
q_max_model = tf.keras.Sequential([
    MeanStdVar(),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                        scale=1e-9 + tf.math.softplus(1.0 * t[...,1:]))),
])
checkpoint_filepath_q_max = './training/aging_model_q_max.ckpt'

# %% Train q_max model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath_q_max,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    verbose=1,
    save_best_only=True)

reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.5, 
    min_lr=1e-6, 
    patience=100,
    min_delta=1e-9,
    verbose=1,
    mode='min')

callbacks=[reduce_lr_on_plateau, model_checkpoint_callback]

q_max_model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-2), loss=negloglik)
q_max_history = q_max_model.fit(X_norm[idx_sort], Y_q_max[idx_sort], epochs=1500, verbose=False, callbacks=callbacks)

fig = plt.figure()
plt.plot(q_max_history.history['loss'])

# %% q_max - Run inference and plots
q_max_model.load_weights(checkpoint_filepath_q_max)
yhat_q_max= q_max_model(X_norm[idx_sort,np.newaxis])

fig = plt.figure()
plt.plot(X,q_max, '.', color='gray')
plt.plot(X[idx_sort], yhat_q_max.quantile(0.025).numpy()[:,0] * max_q_max, 'b--', label='95% CI')
plt.plot(X[idx_sort], yhat_q_max.quantile(0.5).numpy()[:,0] * max_q_max, 'b-', label='Median')
plt.plot(X[idx_sort], yhat_q_max.quantile(0.975).numpy()[:,0] * max_q_max, 'b--')
plt.grid()
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()

# %% q_max - ensemble plot
yhats = [q_max_model(X_norm[idx_sort,np.newaxis]) for _ in range(100)]

avgm_q_max = np.zeros_like(X)
avlb_q_max = np.zeros_like(X)
avup_q_max = np.zeros_like(X)

fig = plt.figure()
plt.plot(X,q_max, '.', color='gray')
for i, yhat in enumerate(yhats):
    m = np.squeeze(yhat.quantile(0.5)) * max_q_max
    # s = np.squeeze(yhat.stddev()) * max_q_max
    lb = np.squeeze(yhat.quantile(0.025)) * max_q_max
    ub = np.squeeze(yhat.quantile(0.975)) * max_q_max

    if i < 15:
        plt.plot(X[idx_sort], m, '-b', label='ensemble median' if i == 0 else None, linewidth=1.)
        plt.plot(X[idx_sort], lb, '--b', linewidth=0.5, label='ensemble 95% CI' if i == 0 else None)
        plt.plot(X[idx_sort], ub, '--b', linewidth=0.5)
    avgm_q_max += m
    avlb_q_max += lb
    avup_q_max += ub

avgm_q_max /= len(yhats)
avlb_q_max /= len(yhats)
avup_q_max /= len(yhats)

plt.plot(X[idx_sort], avgm_q_max, 'r', label='overall median', linewidth=1.)
plt.plot(X[idx_sort], avlb_q_max, '--r', label='overall 95% CI', linewidth=1.)
plt.plot(X[idx_sort], avup_q_max, '--r', linewidth=1.)
plt.grid()
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()


# %% Model for R_0
R_0_model = tf.keras.Sequential([
    MeanStdVar(),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                        scale=1e-11 + tf.math.softplus(1.0 * t[...,1:]))),
])

checkpoint_filepath_R_0 = './training/aging_model_R_0.ckpt'

# %% Train R_0 model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath_R_0,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    verbose=1,
    save_best_only=True)

reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.75, 
    min_lr=1e-6, 
    patience=150,
    min_delta=1e-9,
    verbose=1,
    mode='min')

callbacks=[reduce_lr_on_plateau, model_checkpoint_callback]

R_0_model.compile(optimizer=tf.optimizers.Adam(learning_rate=4e-2), loss=negloglik)
history_R_0 = R_0_model.fit(X_norm[idx_sort], Y_R_0[idx_sort], epochs=4000, verbose=False, callbacks=callbacks)

fig = plt.figure()
plt.plot(history_R_0.history['loss'])

# %% R_0 inference and plot
R_0_model.load_weights(checkpoint_filepath_R_0)
yhat = R_0_model(X_norm[idx_sort,np.newaxis])

fig = plt.figure()
plt.plot(X, R_0, '.', color='gray')
plt.plot(X[idx_sort], yhat.quantile(0.025).numpy()[:,0] * max_R_0, 'b--', label='95% CI')
plt.plot(X[idx_sort], yhat.quantile(0.5).numpy()[:,0] * max_R_0, 'b-', label='Median')
plt.plot(X[idx_sort], yhat.quantile(0.975).numpy()[:,0] * max_R_0, 'b--')
plt.grid()
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$R_0$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()

# %% R_0 ensemble plot
yhats = [R_0_model(X_norm[idx_sort,np.newaxis]) for _ in range(100)]
avgm = np.zeros_like(X)

fig = plt.figure()
plt.plot(X, R_0, '.', color='gray')
for i, yhat in enumerate(yhats):
    m = np.squeeze(yhat.quantile(0.5)) * max_R_0

    lb = np.squeeze(yhat.quantile(0.025)) * max_R_0
    ub = np.squeeze(yhat.quantile(0.975)) * max_R_0

    if i < 15:
        plt.plot(X[idx_sort], m, '-b', label='ensemble median' if i == 0 else None, linewidth=1.)
        plt.plot(X[idx_sort], lb, '--b', linewidth=0.5, label='ensemble 95% CI' if i == 0 else None)
        plt.plot(X[idx_sort], ub, '--b', linewidth=0.5)
    avgm += m

plt.plot(X[idx_sort], avgm/len(yhats), 'r', label='overall median', linewidth=4)
plt.grid()
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$R_0$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()


# %% data for testing
test_idx = np.array(batt_index) == 2

X_test = X_all[test_idx]
q_max_test = q_max_all[test_idx]
R_0_test = R_0_all[test_idx]

idx_sort_test = np.argsort(X_test)

max_q_max_test = np.max(q_max)
Y_q_max_test = q_max_test / max_q_max_test

max_X_test = np.max(X)
X_norm_test = X_test / max_X_test

X_norm_test_ext = np.concatenate([X_norm_test, np.linspace(np.max(X_norm_test), 1.0, 8)])
X_test_ext = X_norm_test_ext * max_X_test

idx_sort_test_ext = np.argsort(X_test_ext)

# max_R_0 = np.max(R_0)
Y_R_0_test = R_0_test / max_R_0

# %% update dist
q_max_model.load_weights(checkpoint_filepath_q_max)

# set the learned posterior as new prior
q_max_model.layers[0].set_dist_weights(
    q_max_model.layers[0].get_dist_weights('_posterior'), 
    dist='_prior'
)
# set prior as not trainable
q_max_model.layers[0].set_dist_trainable(False, '_prior')

q_max_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), loss=negloglik)
q_max_model.summary()

checkpoint_filepath_q_max_test = './training/aging_model_q_max_test.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath_q_max_test,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    verbose=1,
    save_best_only=True)

reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.95, 
    min_lr=1e-6, 
    patience=100,
    min_delta=1e-9,
    verbose=1,
    mode='min')

callbacks=[reduce_lr_on_plateau, model_checkpoint_callback]

points_add = 50
q_max_history = q_max_model.fit(X_norm_test[:points_add], Y_q_max_test[:points_add], epochs=2000, callbacks=callbacks, verbose=False)

# %% q_max - Run inference and plots
q_max_model.load_weights(checkpoint_filepath_q_max_test)
yhats = [q_max_model(X_norm_test_ext[:,np.newaxis]) for _ in range(100)]

avgm = np.zeros_like(X_test_ext)
avlb = np.zeros_like(X_test_ext)
avup = np.zeros_like(X_test_ext)

fig = plt.figure()
plt.plot(X,q_max, '.', color='gray')

plt.plot(X[idx_sort], avgm_q_max, '-', label='Median - Base', color='gray')
plt.plot(X[idx_sort], avlb_q_max, '--', label='95% CI - Base', color='gray')
plt.plot(X[idx_sort], avup_q_max, '--', color='gray')

plt.plot(X_test,q_max_test, 'o', color='black')
plt.plot(X_test[:points_add],q_max_test[:points_add], 'o', color='blue')

for i, yhat in enumerate(yhats):
    m = np.squeeze(yhat.quantile(0.5)) * max_q_max
    # s = np.squeeze(yhat.stddev()) * max_q_max
    lb = np.squeeze(yhat.quantile(0.025)) * max_q_max
    ub = np.squeeze(yhat.quantile(0.975)) * max_q_max

    if i < 15:
        plt.plot(X_test_ext, m, '-b', label='ensemble median' if i == 0 else None, linewidth=1.)
        plt.plot(X_test_ext, lb, '--b', linewidth=0.5, label='ensemble 95% CI' if i == 0 else None)
        plt.plot(X_test_ext, ub, '--b', linewidth=0.5)
    avgm += m
    avlb += lb
    avup += ub

avgm /= len(yhats)
avlb /= len(yhats)
avup /= len(yhats)

plt.plot(X_test_ext, avgm, 'r', label='overall median', linewidth=1.)
plt.plot(X_test_ext, avlb, '--r', label='overall 95% CI', linewidth=1.)
plt.plot(X_test_ext, avup, '--r', linewidth=1.)
plt.grid()
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()


# %%

plt.show()