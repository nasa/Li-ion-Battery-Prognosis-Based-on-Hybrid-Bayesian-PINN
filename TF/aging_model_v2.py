# %% imports
import numpy as np
import math
from time import time
import argparse
from mpl_toolkits.mplot3d import Axes3D
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
test_idx = np.array(batt_index) == 5
# q_max_idx = np.array(batt_index) != 3
X = X_all[q_max_idx]
q_max = q_max_all[q_max_idx]
R_0 = R_0_all[q_max_idx]

idx_sort = np.argsort(X)
max_q_max = np.max(q_max)
Y_q_max = q_max[idx_sort] / max_q_max
Y_q_max -= 1

Y_q_max += 0.02881683

max_X = np.max(X)
X_norm = X[idx_sort] / max_X

max_R_0 = np.max(R_0)
Y_R_0 = R_0[idx_sort] / max_R_0
Y_R_0 = np.min(Y_R_0) - Y_R_0


X_test = X_all[test_idx]
q_max_test = q_max_all[test_idx]
R_0_test = R_0_all[test_idx]

idx_sort_test = np.argsort(X_test)

Y_q_max_test = q_max_test[idx_sort_test] / max_q_max
Y_q_max_test -= 1

Y_q_max_test += 0.02881683

max_X_test = np.max(X)
X_norm_test = X_test[idx_sort_test] / max_X_test

X_norm_test_ext = np.concatenate([X_norm_test, np.linspace(np.max(X_norm_test), 1.0, 8)])
X_test_ext = X_norm_test_ext * max_X_test


Y_R_0_test = R_0_test[idx_sort_test] / max_R_0
Y_R_0_test = np.min(Y_R_0_test) - Y_R_0_test

# %% plot prior dist

prior_loc = [0.0,0.0,0.0,0.0]
prior_scale = [2.0, 2.0, 2.0, 2.0]

mu_k1, mu_k2, mu_b1, mu_b2 = prior_loc
s_k1, s_k2, s_b1, s_b2 = prior_scale

plt_X = np.linspace(mu_k1 - 3*s_k1, mu_k1+3*s_k1,100)
plt_Y = np.linspace(mu_b1 - 3*s_b1, mu_b1+3*s_b1,100)

# limit axis to zoom on later dist
# plt_X = np.linspace(-0.65, -0.35 ,100)
# plt_Y = np.linspace(-0.06, 0.01 ,100)

plt_X, plt_Y = np.meshgrid(plt_X, plt_Y)

plt_Z = np.zeros_like(plt_X)

batch_xy = np.vstack([np.ravel(plt_X),np.ravel(plt_Y)]).T

Z_temp = tfd.MultivariateNormalDiag(loc=[mu_k1, mu_b1], scale_diag=[s_k1,s_b1]).prob(batch_xy).numpy()

plt_Z = Z_temp.reshape((100,100)) / np.max(Z_temp)

fig = plt.figure('surface')
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(plt_X, plt_Y, plt_Z, cmap='coolwarm')
# surf = ax.plot_wireframe(plt_X, plt_Y, plt_Z, color='gray', rstride=5, cstride=5)
surf = plt.contourf(plt_X, plt_Y, plt_Z)
# fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('k1')
ax.set_ylabel('b1')
# ax.set_zlabel('pdf')


# %%
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: 
        # tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                        scale=1e-9 + tf.nn.softplus(c + t[..., n:])),
            # reinterpreted_batch_ndims=1)
            ),
    ])

# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def get_prior_fn(loc, scale):
    def prior(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            # tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: 
            # tfd.Independent(
                tfd.Normal(loc=loc , scale=scale),
                # reinterpreted_batch_ndims=1)
            ),
        ])
    return prior

model = tf.keras.Sequential([
    tfp.layers.DenseVariational(1+1, posterior_mean_field, get_prior_fn(prior_loc, prior_scale), kl_weight=1/X_norm.shape[0], use_bias=True),
    tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-9 + tf.math.softplus(1.0 * t[...,1:]))),
])

negloglik = lambda y, rv_y: -rv_y.log_prob(y)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.04), loss=negloglik)

model.build(input_shape=(X_norm.shape[0],1))
# model.set_weights([np.array([[-1.0,1.0]])])

reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.5, 
    min_lr=1e-6, 
    patience=200,
    min_delta=1e-9,
    verbose=1,
    mode='min')

callbacks=[reduce_lr_on_plateau]

history = model.fit(X_norm, Y_q_max, epochs=1000, callbacks=callbacks, verbose=2)

# # %%
# dist = model(X_norm[:,np.newaxis])
# loc = dist.loc.numpy()
# scale = dist.scale.numpy()

# fig = plt.figure()
# plt.plot(X_norm, Y_q_max, '.', color='gray')
# plt.plot(X_norm, loc, 'b-')
# plt.plot(X_norm, loc + 2*scale, 'b--')
# plt.plot(X_norm, loc - 2*scale, 'b--')
# plt.grid()

# %%
yhats = [model(X_norm[:,np.newaxis]) for _ in range(100)]
fig = plt.figure()
plt.plot(X_norm, Y_q_max, '.', color='gray')
avgm_base = np.zeros_like(X_norm)
avlb_base = np.zeros_like(X_norm)
avup_base = np.zeros_like(X_norm)
for i, yhat in enumerate(yhats):
    m = np.squeeze(yhat.loc)
    s = np.squeeze(yhat.scale)
    lb = np.squeeze(m - 2*s)
    ub = np.squeeze(m + 2*s)

    avgm_base += m
    avlb_base += lb
    avup_base += ub

avgm_base /= len(yhats)
avlb_base /= len(yhats)
avup_base /= len(yhats)

plt.plot(X_norm, avgm_base, 'r', label='overall median', linewidth=1.)
plt.plot(X_norm, avlb_base, '--r', label='overall 2 * $\sigma$', linewidth=1.)
plt.plot(X_norm, avup_base, '--r', linewidth=1.)
plt.grid()


# %% plot base posterior
post_weights = model.layers[0]._posterior.get_weights()[0]

#posterior weights with bias and 2 units
mu_k1, mu_k2, mu_b1, mu_b2 = post_weights[:4]

c = np.log(np.expm1(1.))
s_k1, s_k2, s_b1, s_b2 = (1e-9 + tf.math.softplus(c + post_weights[4:]).numpy())

plt_X = np.linspace(mu_k1 - 3*s_k1, mu_k1+3*s_k1,100)
plt_Y = np.linspace(mu_b1 - 3*s_b1, mu_b1+3*s_b1,100)
plt_X, plt_Y = np.meshgrid(plt_X, plt_Y)

plt_Z = np.zeros_like(plt_X)

batch_xy = np.vstack([np.ravel(plt_X),np.ravel(plt_Y)]).T

Z_temp = tfd.MultivariateNormalDiag(loc=[mu_k1, mu_b1], scale_diag=[s_k1,s_b1]).prob(batch_xy).numpy()

plt_Z = Z_temp.reshape((100,100)) / np.max(Z_temp)

# fig = plt.figure('surface')
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(plt_X, plt_Y, plt_Z, cmap='coolwarm')
surf = ax.plot_wireframe(plt_X, plt_Y, plt_Z, color='blue')
# fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('k1')
ax.set_ylabel('b1')
ax.set_zlabel('pdf')


# %%
prior_fn = get_prior_fn([mu_k1,mu_k2,mu_b1,mu_b2], [s_k1, s_k2, s_b1, s_b2])

model_test = tf.keras.Sequential([
    tfp.layers.DenseVariational(1+1, posterior_mean_field, prior_fn, kl_weight=1/X_norm.shape[0], use_bias=True),
    tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],
                           scale=1e-9 + tf.math.softplus(1.0 * t[...,1:]))),
])

model_test.compile(optimizer=tf.optimizers.Adam(learning_rate=0.04), loss=negloglik)
model_test.build(input_shape=(X_norm_test.shape[0],1))

# model_test.set_weights([post_weights])

checkpoint_filepath = './training/aging_model_v2_q_max_batt6.ckpt'

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
    patience=300,
    min_delta=1e-9,
    verbose=1,
    mode='min')

callbacks=[reduce_lr_on_plateau,model_checkpoint_callback]

history = model_test.fit(X_norm_test, Y_q_max_test, epochs=5000, callbacks=callbacks, verbose=False)

# %% infer and plot test model
model_test.load_weights(checkpoint_filepath)

# dist = model_test(X_norm_test[:,np.newaxis])
# loc = dist.loc.numpy()
# scale = dist.scale.numpy()

# fig = plt.figure()
# plt.plot(X_norm, Y_q_max, '.', color='gray')
# plt.plot(X_norm_test, Y_q_max_test, '.', color='black')
# plt.plot(X_norm_test, loc, 'b-')
# plt.plot(X_norm_test, loc + 2*scale, 'b--')
# plt.plot(X_norm_test, loc - 2*scale, 'b--')
# plt.grid()

yhats = [model_test(X_norm_test[:,np.newaxis]) for _ in range(100)]
fig = plt.figure()
plt.plot(X_norm, Y_q_max, '.', color='gray')
plt.plot(X_norm, avgm_base, '-', label='base median', linewidth=1., color='blue')
plt.plot(X_norm, avlb_base, '--', label='base 2 * $\sigma$', linewidth=1., color='blue')
plt.plot(X_norm, avup_base, '--', linewidth=1., color='blue')


plt.plot(X_norm_test, Y_q_max_test, '.', color='black')
avgm = np.zeros_like(X_norm_test)
avlb = np.zeros_like(X_norm_test)
avup = np.zeros_like(X_norm_test)
for i, yhat in enumerate(yhats):
    m = np.squeeze(yhat.loc)
    s = np.squeeze(yhat.scale)
    lb = np.squeeze(m - 2*s)
    ub = np.squeeze(m + 2*s)

    avgm += m
    avlb += lb
    avup += ub

avgm /= len(yhats)
avlb /= len(yhats)
avup /= len(yhats)

plt.plot(X_norm_test, avgm, 'r', label='overall median', linewidth=1.)
plt.plot(X_norm_test, avlb, '--r', label='overall 2 * $\sigma$', linewidth=1.)
plt.plot(X_norm_test, avup, '--r', linewidth=1.)
plt.grid()

# %%
post_weights_test = model_test.layers[0]._posterior.get_weights()[0]

#posterior weights with bias and 2 units
mu_k1_test, mu_k2_test, mu_b1_test, mu_b2_test = post_weights_test[:4]

c = np.log(np.expm1(1.))
s_k1_test, s_k2_test, s_b1_test, s_b2_test = (1e-9 + tf.math.softplus(c + post_weights_test[4:]).numpy())

# mu_k1 = 0.
# mu_b1 = 0.
# s_k1 = 2.
# s_b1 = 2.

plt_X = np.linspace(mu_k1_test - 3*s_k1_test, mu_k1_test+3*s_k1_test,100)
plt_Y = np.linspace(mu_b1_test - 3*s_b1_test, mu_b1_test+3*s_b1_test,100)

plt_X, plt_Y = np.meshgrid(plt_X, plt_Y)

plt_Z = np.zeros_like(plt_X)

batch_xy = np.vstack([np.ravel(plt_X),np.ravel(plt_Y)]).T

Z_temp = tfd.MultivariateNormalDiag(loc=[mu_k1_test, mu_b1_test], scale_diag=[s_k1_test,s_b1_test]).prob(batch_xy).numpy()

plt_Z = Z_temp.reshape((100,100)) / np.max(Z_temp)

# fig = plt.figure('surface')
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(plt_X, plt_Y, plt_Z, cmap='viridis')
surf = ax.plot_wireframe(plt_X, plt_Y, plt_Z, color='red')
# fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('k1')
ax.set_ylabel('b1')
ax.set_zlabel('pdf')

# ax.set_xlim([-0.65,-0.35])
# ax.set_ylim([-0.06,0.01])


# %%




# X_dist = []
# Y_q_dist = []
# Y_R_dist = []


# skip = [2,5]
# max_num_dist = max([len(s) for i,s in enumerate(sizes) if i not in skip])
# batt_list = [j for j in range(len(sizes)) if j not in skip]

# for i in np.arange(0,max_num_dist,2):
#     x = []
#     y_q = []
#     y_R = []
#     for batt_i in batt_list:
#         bidx = np.argwhere(np.array(batt_index)==batt_i)[:,0]
#         if i < len(bidx):
#             x.append(X_all[bidx[i]])
#             y_q.append(q_max_all[bidx[i]])
#             y_R.append(R_0_all[bidx[i]])
#     i += 1
#     for batt_i in batt_list:
#         bidx = np.argwhere(np.array(batt_index)==batt_i)[:,0]
#         if i < len(bidx):
#             x.append(X_all[bidx[i]])
#             y_q.append(q_max_all[bidx[i]])
#             y_R.append(R_0_all[bidx[i]])

#     X_dist.append(np.array(x))
#     Y_q_dist.append(np.array(y_q))

# # %%
# fig = plt.figure()
# for i in range(len(X_dist)):
#     plt.plot(X_dist[i], Y_q_dist[i], '.')
# plt.grid()

# %%

# mvn = tfd.MultivariateNormalDiag(
#     loc=loc,
#     scale_diag=scale)

# negloglik = lambda y, rv_y: -rv_y.log_prob(y)

plt.show()