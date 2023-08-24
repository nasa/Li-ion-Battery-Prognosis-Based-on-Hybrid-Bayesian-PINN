# %% imports
import sys
from os import path
import argparse
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

sys.argv = ['']

parser = argparse.ArgumentParser()
parser.add_argument("--save", default=False, action="store_true" , help="Save results")
parser.add_argument("--train", default=False, action="store_true" , help="Force train even if saved results exist")
args = parser.parse_args()

SAVE_DATA_PATH = './training/aging_model_v3.npy'

TRAIN = True
if path.exists(SAVE_DATA_PATH) and not args.train:
    TRAIN = False

# %% load data
# -------------------------------------
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
batt_index = np.array(batt_index)

PWh = [(np.cumsum(power_time[i])/3.6e6) for i in range(len(power_time))]

cum_kWh_ref = []
q_max_ref = []
R_0_ref = []
for i in range(len(init_time)):
    cum_kWh_ref.append(
        np.array([PWh[i][np.argwhere(time_all[i]>=init_time[i][j])[0][0]] for j in range(len(init_time[i]))])
    )
    q_max_ref.append(
        q_max_all[batt_index==i]
    )
    R_0_ref.append(
        R_0_all[batt_index==i]
    )

X_all = np.zeros(q_max_all.shape)
for i in range(len(q_max_all)):
    idx = np.argwhere(time_all[batt_index[i]]>=inputs_time[i,0])[0][0]
    X_all[i] = PWh[batt_index[i]][idx]

X_MAX = max(X_all)
Y_MAX = max(q_max_all)

n_batt = len(q_max_ref)

# n_batt = 1

skip = [3]
X_test = cum_kWh_ref[skip[0]]
Y_test = q_max_ref[skip[0]]

# %% MODELs def and func
# -------------------------------------
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: #tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(t[..., n:])),
        #   reinterpreted_batch_ndims=1)
            ),
    ])

class PriorDist():
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def adjust_loc_scale(self, n):
        if not isinstance(self.loc, (list, tuple, np.ndarray)):
            self.loc = np.ones(n, dtype='float32')*self.loc
        if not isinstance(self.scale, (list, tuple, np.ndarray)):
            self.scale = np.ones(n, dtype='float32')*self.scale

    def prior_fn(self, kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        
        self.adjust_loc_scale(n)

        return tf.keras.Sequential([
            # tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: #tfd.Independent(
                # tfd.Normal(loc=t , scale=self.scale),
                tfd.Normal(loc=self.loc , scale=self.scale),
        #   reinterpreted_batch_ndims=1)
            ),
        ])


class MeanStdVar(tf.keras.layers.Layer):
    def __init__(self, prior_loc=0, prior_scale=2, batch_size=1):
        super(MeanStdVar, self).__init__()
        self.mean = tf.keras.Sequential([
            # tf.keras.layers.Dense(4, activation='elu'),
            # tf.keras.layers.Dense(1, activation='elu'),
            # tfp.layers.DenseVariational(6, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(4, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(2, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size, activation='elu'),
            # tfp.layers.DenseVariational(1, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size),
            tfp.layers.DenseVariational(1, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size)
        ])
        self.std = tf.keras.Sequential([
            tfp.layers.DenseVariational(1, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size)
        ])
    
    def build(self, input_shape):
        super(MeanStdVar, self).__init__()
        # self.mean.build(input_shape=input_shape)
        # self.std.build(input_shape=input_shape)

    def call(self, inputs):
        return tf.keras.layers.concatenate([self.mean(inputs), self.std(inputs)])


def get_model(prior_loc=0, prior_scale=2, batch_size=300):
    model = tf.keras.Sequential([
        MeanStdVar(prior_loc=prior_loc, prior_scale=prior_scale, batch_size=batch_size),
        tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-9 + tf.math.softplus(1 * t[...,1:]))),
    ])

    return model

def train_model(model, X, Y, batt_i, batch_size=32, epochs=6000, init_lr=0.04, loss=None):
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    if loss is None:
        loss = negloglik
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=init_lr), loss=loss)
    model.build(input_shape=(X.shape[0],1))

    # model_test.set_weights([post_weights])

    checkpoint_filepath = './training/aging_model_v3_q_max_batt_{}.ckpt'.format(batt_i+1)

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
        min_lr=1e-5, 
        patience=200,
        min_delta=1e-9,
        verbose=1,
        mode='min')

    callbacks=[reduce_lr_on_plateau,model_checkpoint_callback]

    model.summary()

    print('')
    print('** Fiting batt #{} model for {} epochs **'.format(batt_i+1, epochs))
    history = model.fit(X, Y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=False)

    print('** Final loss for batt #{} model: {} **'.format(batt_i+1, np.min(history.history['loss'])))

    model.load_weights(checkpoint_filepath)

    print('Model Weights:', model.get_weights())

    return history

# %% Create and fit or load saved model for each battery
# -------------------------------------
if TRAIN:
    model_dic_list = []
    for batt_i in range(n_batt):
        model_dic = {
            'batt_i': batt_i
        }
        print('')
        print('* * * Building batt model - {}/{} * * *'.format(batt_i+1, n_batt))

        X = cum_kWh_ref[batt_i]
        X_norm = X / X_MAX
        Y = q_max_ref[batt_i]
        Y_norm = Y / Y_MAX - 1
        # Y_norm = Y / Y_MAX

        # model = get_model(batch_size=X.shape[0])
        model = get_model()

        # fit
        model_dic['history'] = train_model(model, X_norm, Y_norm, batt_i).history
        model_dic['final_loss'] = np.min(model_dic['history']['loss'])

        model_dic['weights'] = model.get_weights()
        model_dic['model'] = model

        num_random_draw = 100
        print('* Getting Avg. mean and std from {} random draw *'.format(num_random_draw))
        
        yhats = [model(X_norm[:,np.newaxis]) for _ in range(num_random_draw)]

        avg_m = np.zeros_like(X)
        avg_s = np.zeros_like(X)
        for i, yhat in enumerate(yhats):
            m = np.squeeze(yhat.loc)
            s = np.squeeze(yhat.scale)
            avg_m += m
            avg_s += s

        avg_m /= len(yhats)
        avg_s /= len(yhats)

        model_dic['avg_m'] = avg_m
        model_dic['avg_s'] = avg_s

        model_dic_list.append(model_dic)
else:
    #load saved data
    model_dic_list = np.load(SAVE_DATA_PATH, allow_pickle=True)

    for batt_i in range(n_batt):
        X = cum_kWh_ref[batt_i]

        model = get_model(batch_size=X.shape[0])
        model.build(input_shape=(X.shape[0],1))
        model.set_weights(model_dic_list[batt_i]['weights'])
        model_dic_list[batt_i]['model'] = model
        

# %%
# fig = plt.figure()
# for batt_i in range(n_batt):
#     fig = plt.figure()
#     X = cum_kWh_ref[batt_i]
#     X_norm = X / X_MAX
#     Y = q_max_ref[batt_i]
#     Y_norm = Y / Y_MAX - 1
#     plt.plot(X_norm, Y_norm, '.', color='C{}'.format(batt_i))
#     plt.plot(X_norm, model_dic_list[batt_i]['avg_m'], '-', linewidth=1., color='C{}'.format(batt_i))
#     plt.plot(X_norm, model_dic_list[batt_i]['avg_m'] - 2*model_dic_list[batt_i]['avg_s'], '--', linewidth=1., color='C{}'.format(batt_i))
#     plt.plot(X_norm, model_dic_list[batt_i]['avg_m'] + 2*model_dic_list[batt_i]['avg_s'], '--', linewidth=1., color='C{}'.format(batt_i))
#     plt.grid()

# %%
for batt_i in range(n_batt):
    fig = plt.figure()
    X = cum_kWh_ref[batt_i]
    X_norm = X / X_MAX
    Y = q_max_ref[batt_i]
    Y_norm = Y / Y_MAX - 1
    plt.plot(X_test, Y_test, 'k.')
    plt.plot(X, Y, '.', color='C{}'.format(batt_i))
    m = (model_dic_list[batt_i]['avg_m']+1)*Y_MAX
    lb = ((model_dic_list[batt_i]['avg_m'] - 2*model_dic_list[batt_i]['avg_s']) +1)*Y_MAX
    ub = ((model_dic_list[batt_i]['avg_m'] + 2*model_dic_list[batt_i]['avg_s']) +1)*Y_MAX
    plt.plot(X, m, '-', linewidth=1., color='C{}'.format(batt_i))
    plt.plot(X, lb, '--', linewidth=1., color='C{}'.format(batt_i))
    plt.plot(X, ub, '--', linewidth=1., color='C{}'.format(batt_i))
    plt.title('Batt #{}'.format(batt_i+1))
    # plt.xlim([0,3.5])
    # plt.ylim([6000,14000])
    plt.grid()


# %%
def get_mu_s_from_model(batt_i, X):
    # model = get_model(batch_size=X.shape[0])
    # model.build(input_shape=(X.shape[0],1))
    # model.set_weights(model_dic_list[batt_i]['weights'])

    model = model_dic_list[batt_i]['model']

    num_random_draw = 100
    yhats = [model(X[:,np.newaxis]/X_MAX) for _ in range(num_random_draw)]

    avg_m = np.zeros_like(X)
    avg_s = np.zeros_like(X)
    for i, yhat in enumerate(yhats):
        m = np.squeeze(yhat.loc)
        s = np.squeeze(yhat.scale)
        avg_m += m
        avg_s += s

    avg_m /= len(yhats)
    avg_s /= len(yhats)

    return avg_m, avg_s

# %% Plot, all dist

fig = plt.figure()
plt.plot(X_test, Y_test, '.', color='black')

for batt_i in range(n_batt):
    if batt_i in skip:
        continue

    X = cum_kWh_ref[batt_i]
    Y = q_max_ref[batt_i]

    m = (model_dic_list[batt_i]['avg_m']+1)*Y_MAX
    lb = ((model_dic_list[batt_i]['avg_m'] - 2*model_dic_list[batt_i]['avg_s']) +1)*Y_MAX
    ub = ((model_dic_list[batt_i]['avg_m'] + 2*model_dic_list[batt_i]['avg_s']) +1)*Y_MAX

    plt.fill_between(X, ub, lb, alpha=0.2, color='C{}'.format(batt_i))
    plt.plot(X, Y, '.', color='C{}'.format(batt_i))
    plt.plot(X, m, '-', color='C{}'.format(batt_i))

plt.grid()
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# %%

fig = plt.figure()
plt.plot(X_test, Y_test, '.', color='black')

like = np.zeros((X_test.shape[0],n_batt))

mean_models = np.zeros((X_test.shape[0],n_batt))
std_models = np.zeros((X_test.shape[0],n_batt))

for batt_i in range(n_batt):
    if batt_i in skip:
        continue
    
    m_avg, s = get_mu_s_from_model(batt_i, X_test)

    # lb = (m - 2*s)
    # ub = (m + 2*s)

    # max_Y = max(q_max_ref[batt_i])

    m = (m_avg+1)*Y_MAX

    lb = ((m_avg - 2*s) +1)*Y_MAX
    ub = ((m_avg + 2*s) +1)*Y_MAX

    s = (ub-lb)/4

    mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)

    like[:,batt_i] = mvn.log_prob(Y_test)

    mean_models[:,batt_i]  = m
    std_models[:,batt_i]  = s

    plt.fill_between(X_test, ub, lb, alpha=0.2, color='C{}'.format(batt_i))
    plt.plot(X_test, m, '-', color='C{}'.format(batt_i))
    # plt.plot(X_test, lb, '--', linewidth=1., color='C{}'.format(batt_i))
    # plt.plot(X_test, ub, '--', linewidth=1., color='C{}'.format(batt_i))
    # m = model_dic_list[batt_i]['avg_m']
    # s = model_dic_list[batt_i]['avg_s']
plt.grid()
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


# %%
idx = [i for i in range(n_batt) if i not in skip]
# like = like[:,idx]
w = np.sum(like[:,idx], axis=0)
w_norm = (w-w.min())/(w.max()-w.min())
w_norm = w_norm / np.sum(w_norm)

w_norm_full = np.zeros(n_batt)
w_norm_full[idx] = w_norm
# %%
n_samples = np.round(1000*w_norm_full)
# Y_samples = np.zeros((X_test.shape[0],np.sum(n_samples, dtype=int)))
Y_samples = []

for batt_i in range(n_batt):
    if batt_i in skip:
        continue
    
    m = mean_models[:,batt_i]
    s = std_models[:,batt_i]


    mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)

    Y_samples.append(mvn.sample(n_samples[batt_i]))

Y_samples = np.vstack(Y_samples).T

# %%
Y_prcntl = np.percentile(Y_samples, [2.5, 50.0, 97.5], axis=1)

fig = plt.figure()
plt.plot(X_test, Y_test, 'ok')
plt.plot(X_test, Y_prcntl[1], '-b', linewidth=1.)
plt.plot(X_test, Y_prcntl[0], '--b', linewidth=1.)
plt.plot(X_test, Y_prcntl[2], '--b', linewidth=1.)
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()

# %%

# %% MSE plot
mse = np.zeros(n_batt)
for batt_i in range(n_batt):
    fig = plt.figure()
    # fig = plt.figure('Batt #{}'.format(n_batt))
    # fig.clf()
    X = cum_kWh_ref[batt_i]
    X_norm = X / X_MAX
    Y = q_max_ref[batt_i]
    Y_norm = Y / Y_MAX - 1
    plt.plot(X_test, Y_test, 'k.')
    plt.plot(X, Y, '.', color='C{}'.format(batt_i))

    m_avg, s_avg = get_mu_s_from_model(batt_i, X_test)

    m = (m_avg+1)*Y_MAX
    lb = ((m_avg - 2*s_avg) +1)*Y_MAX
    ub = ((m_avg + 2*s_avg) +1)*Y_MAX

    s = (ub-lb)/4

    mse[batt_i] = np.mean((m - Y_test)**2)

    plt.plot(X_test, m, '-', linewidth=1., color='C{}'.format(batt_i))
    plt.plot(X_test, lb, '--', linewidth=1., color='C{}'.format(batt_i))
    plt.plot(X_test, ub, '--', linewidth=1., color='C{}'.format(batt_i))
    plt.title('Batt #{:} - MSE:{:.2e}'.format(batt_i+1, mse[batt_i]))
    # plt.xlim([0,3.5])
    # plt.ylim([6000,14000])
    plt.xlabel('Cumulative Energy (kWh)')
    plt.ylabel(r'$q_{MAX}$')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.grid()

    if args.save:
        fig.savefig('./figures/ensamble_mse_batt_{}.png'.format(batt_i))

# %%
Cinv = np.linalg.inv(np.diag(mse[idx]))
e = np.ones((n_batt-1,1))
lamb = 2/(e.T@Cinv@e)
w_mse = (lamb * (Cinv@e)/2)
w_mse_full = np.zeros(n_batt)
w_mse_full[idx] = w_mse[:,0]

# %%
n_samples_mse = np.round(1000*w_mse_full)
# Y_samples = np.zeros((X_test.shape[0],np.sum(n_samples, dtype=int)))
Y_samples_mse = []

for batt_i in range(n_batt):
    if batt_i in skip:
        continue
    
    m = mean_models[:,batt_i]
    s = std_models[:,batt_i]


    mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)

    Y_samples_mse.append(mvn.sample(n_samples_mse[batt_i]))

Y_samples_mse = np.vstack(Y_samples_mse).T

# %%
mix = tfd.Mixture(
    cat=tfd.Categorical(probs=w_mse[:,0]),
    components=[
        tfd.MultivariateNormalDiag(loc=mean_models[:,batt_i], scale_diag=std_models[:,batt_i])
        for batt_i in idx
        ]
)

Y_samples_mix = mix.sample(1000).numpy()
Y_prcntl_mix = np.percentile(Y_samples_mix, [2.5, 50.0, 97.5], axis=0)
# %%

Y_prcntl_mse = np.percentile(Y_samples_mse, [2.5, 50.0, 97.5], axis=1)

fig = plt.figure()
plt.plot(X_test, Y_test, 'ok')

plt.plot(X_test, Y_prcntl[1], '-', color='C0', linewidth=1., label='Likelihood')
plt.plot(X_test, Y_prcntl[0], '--', color='C0', linewidth=1.)
plt.plot(X_test, Y_prcntl[2], '--', color='C0', linewidth=1.)

plt.plot(X_test, Y_prcntl_mse[1], '-', color='C1', linewidth=1., label='MSE')
plt.plot(X_test, Y_prcntl_mse[0], '--', color='C1', linewidth=1.)
plt.plot(X_test, Y_prcntl_mse[2], '--', color='C1', linewidth=1.)

plt.plot(X_test, Y_prcntl_mix[1], '-', color='C2', linewidth=1., label='MIX MSE')
plt.plot(X_test, Y_prcntl_mix[0], '--', color='C2', linewidth=1.)
plt.plot(X_test, Y_prcntl_mix[2], '--', color='C2', linewidth=1.)
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()

# %% compare mse with likelihood
fig, ax = plt.subplots()
ind = np.arange(n_batt)+1
width = 0.35
ax.bar(ind, w_norm_full, width, label='Likelihood')
ax.bar(ind + width, w_mse_full, width, label='MSE')

ax.set_xticks(ind + width / 2)
ax.set_xticklabels(ind)
ax.set_xlabel('Battery #')
ax.set_ylabel('weight')
ax.legend()

# %%

# %% test time
num_test_points = 6
mse_test = np.zeros(n_batt)

mean_models_test = np.zeros((X_test[:num_test_points].shape[0],n_batt))
std_models_test = np.zeros((X_test[:num_test_points].shape[0],n_batt))

for batt_i in range(n_batt):
    X = cum_kWh_ref[batt_i]
    Y = q_max_ref[batt_i]

    m_avg, s_avg = get_mu_s_from_model(batt_i, X_test[:num_test_points])

    m = (m_avg+1)*Y_MAX
    lb = ((m_avg - 2*s_avg) +1)*Y_MAX
    ub = ((m_avg + 2*s_avg) +1)*Y_MAX

    s = (ub-lb)/4

    mean_models_test[:,batt_i]  = m
    std_models_test[:,batt_i]  = s

    mse_test[batt_i] = np.mean((m - Y_test[:num_test_points])**2)

# %%
Cinv_test = np.linalg.inv(np.diag(mse_test[idx]))
e_test = np.ones((n_batt-1,1))
lamb_test = 2/(e_test.T@Cinv_test@e_test)
w_mse_test = (lamb_test * (Cinv_test@e_test)/2)
w_mse_full_test = np.zeros(n_batt)
w_mse_full_test[idx] = w_mse_test[:,0]

# %%
n_samples_mse_test = np.round(1000*w_mse_full_test)
# Y_samples = np.zeros((X_test.shape[0],np.sum(n_samples, dtype=int)))
Y_samples_mse_test = []
X_test_ext = np.linspace(0, 3.5, 100)

for batt_i in range(n_batt):
    if batt_i in skip:
        continue
    
    m_avg, s_avg = get_mu_s_from_model(batt_i, X_test_ext)

    m = (m_avg+1)*Y_MAX
    lb = ((m_avg - 2*s_avg) +1)*Y_MAX
    ub = ((m_avg + 2*s_avg) +1)*Y_MAX

    s = (ub-lb)/4


    mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)

    Y_samples_mse_test.append(mvn.sample(n_samples_mse_test[batt_i]))

Y_samples_mse_test = np.vstack(Y_samples_mse_test).T

# %%

Y_prcntl_mse_test = np.percentile(Y_samples_mse_test, [2.5, 50.0, 97.5], axis=1)

fig = plt.figure()
plt.plot(X_test, Y_test, 'o', color='gray')
plt.plot(X_test[:num_test_points], Y_test[:num_test_points], 'ok')

plt.plot(X_test_ext, Y_prcntl_mse_test[1], '-', color='C1', linewidth=1., label='MSE')
plt.plot(X_test_ext, Y_prcntl_mse_test[0], '--', color='C1', linewidth=1.)
plt.plot(X_test_ext, Y_prcntl_mse_test[2], '--', color='C1', linewidth=1.)
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()

# %%
fig, ax = plt.subplots()
ind = np.arange(n_batt)+1
width = 0.35
ax.bar(ind, w_mse_full, width, label='MSE all', color='gray')
ax.bar(ind + width, w_mse_full_test, width, label='MSE test pts', color='C1')

ax.set_xticks(ind + width / 2)
ax.set_xticklabels(ind)
ax.legend()

# %% fit model into ensemble to get prior

# X = np.hstack([X_test_ext for _ in range(Y_samples_mse_test.shape[1])])
X = np.hstack([X_test_ext for _ in range(3)])
X_norm = X / X_MAX
# Y = np.hstack([Y_samples_mse_test[:,i] for i in range(Y_samples_mse_test.shape[1])])
y_mean = np.mean(Y_samples_mse_test, axis=1)
y_std = np.std(Y_samples_mse_test, axis=1)
Y = np.hstack([y_mean,y_mean+y_std,y_mean-y_std])
Y_norm = Y / Y_MAX - 1

model = get_model(prior_loc=0, prior_scale=2, batch_size=10000)

def loss_fn(y_true,y_dist):
    # tf.print('y_true.shape', y_true.shape)
    # tf.print('y_dist.quantile.shape', y_dist.quantile(0.5).shape)
    return tf.keras.losses.MSE(y_true[:,0], y_dist.quantile(0.025)) + tf.keras.losses.MSE(y_true[:,1], y_dist.quantile(0.5))


# fit
history = train_model(model, X_norm, Y_norm, 100, batch_size=16, epochs=2000).history

# %%
model = get_model(prior_loc=0, prior_scale=2, batch_size=10000)
model.build(input_shape=(X_norm.shape[0],1))
model.set_weights(prior_weights)
# %%
num_random_draw = 100
yhats_dist = [model(X_test_ext[:,np.newaxis]/X_MAX) for _ in range(num_random_draw)]
yhats = np.vstack([dist.sample(num_random_draw).numpy() for dist in yhats_dist])
yhats = (yhats+1)*Y_MAX
# %%
Y_prcntl_prior = np.percentile(yhats[...,0], [2.5, 50.0, 97.5], axis=0)

fig = plt.figure()
plt.plot(X_test, Y_test, 'o', color='gray')
plt.plot(X_test[:num_test_points], Y_test[:num_test_points], 'ok')

plt.plot(X_test_ext, Y_prcntl_mse_test[1], '-', color='C1', linewidth=1., label='MSE')
plt.plot(X_test_ext, Y_prcntl_mse_test[0], '--', color='C1', linewidth=1.)
plt.plot(X_test_ext, Y_prcntl_mse_test[2], '--', color='C1', linewidth=1.)

plt.plot(X_test_ext, Y_prcntl_prior[1], '-', color='C2', linewidth=1., label='Prior')
plt.plot(X_test_ext, Y_prcntl_prior[0], '--', color='C2', linewidth=1.)
plt.plot(X_test_ext, Y_prcntl_prior[2], '--', color='C2', linewidth=1.)
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()


# %%
class DistFromMix(tfd.Normal):
    _data = {
        'mix': None,
        'cat_prob': None,
        'layers_size': None,
        'loc': [],
        'scale': [],
        'draw': None
    }
    def __init__(self, cat_prob, idx, layer_i, total_layers=4, **kwargs):
        self.n = len(model_dic_list[0]['weights'][layer_i])//2
        self.layer_i = layer_i

        if layer_i==0:
            self._data['cat_prob'] = cat_prob
            self._data['loc'] = []
            self._data['scale'] = []

            base = 0
            self._data['layers_size'] = []
            self._data['layers_slices'] = []

            for l_i in range(total_layers):
                size = len(model_dic_list[0]['weights'][l_i])//2
                start = base
                end = start+size
                base = end
                self._data['layers_size'].append(size)
                self._data['layers_slices'].append((start,end))

            loc_size = np.sum(self._data['layers_size'])

            for batt_i in idx:
                loc = np.zeros(loc_size, dtype='float32')
                scale = np.zeros(loc_size, dtype='float32')
                for l_i in range(total_layers):
                    m,s = np.split(model_dic_list[batt_i]['weights'][l_i], 2)
                    i,j = self._data['layers_slices'][l_i]

                    loc[i:j] = m
                    scale[i:j] = s

                self._data['loc'].append(loc)
                self._data['scale'].append(scale)

            self._data['mix'] = tfd.Mixture(
                # cat=tfd.Categorical(probs=np.tile(self._data['cat_prob'],loc_size).T),
                cat=tfd.Categorical(probs=self._data['cat_prob'][:,0]),
                components=[
                    tfd.MultivariateNormalDiag(loc=self._data['loc'][i], scale_diag=1e-5 + tf.nn.softplus(self._data['scale'][i]))
                    for i in range(len(idx))
                    ]
            )
        
        i,j = self._data['layers_slices'][self.layer_i]
        self.mix = tfd.Mixture(
                # cat=tfd.Categorical(probs=np.tile(self._data['cat_prob'],self.n).T),
                cat=tfd.Categorical(probs=self._data['cat_prob'][:,0]),
                components=[
                    tfd.MultivariateNormalDiag(loc=self._data['loc'][k][i:j], scale_diag=1e-5 + tf.nn.softplus(self._data['scale'][k][i:j]))
                    for k in range(len(idx))
                    ]
            )
        self.mix_list = [tfd.Normal(
                loc=self._data['loc'][k][i:j], scale=1e-5 + tf.nn.softplus(self._data['scale'][k][i:j])
            ) for k in range(len(idx)) ]
        
        super(DistFromMix, self).__init__(loc=self._data['mix'].mean()[i:j], scale=self._data['mix'].stddev()[i:j] )

    def get_draw(self, dist):
        if self.layer_i==0:
            self._data['draw'] = self._data['mix'].sample()

        i,j = self._data['layers_slices'][self.layer_i]
        return self._data['draw'][i:j]

    def log_prob(self, w):
        ret = tf.reduce_sum(tf.stack([dist.log_prob(w) for dist in self.mix_list]) * self._data['cat_prob'], axis=0)
        # tf.print(ret)
        return ret


    def fn(self, kernel_size, bias_size=0, dtype=None):
        dist_self = self
        
        return tf.keras.Sequential([
            tfp.layers.DistributionLambda(lambda t: dist_self.mix,
            # convert_to_tensor_fn=dist_self.get_draw
            )
        ])

# %%
class MeanStdVarPostMix(tf.keras.layers.Layer):
    def __init__(self, cat_prob, idx, prior_loc=0, prior_scale=2, batch_size=1):
        super(MeanStdVarPostMix, self).__init__()
        self.mean = tf.keras.Sequential([
            tfp.layers.DenseVariational(4, DistFromMix(cat_prob, idx, 0).fn, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(2, DistFromMix(cat_prob, idx, 1).fn, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(1, DistFromMix(cat_prob, idx, 2).fn, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size)
        ])
        self.std = tf.keras.Sequential([
            tfp.layers.DenseVariational(1, DistFromMix(cat_prob, idx, 3).fn, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size)
        ])
    
    def build(self, input_shape):
        super(MeanStdVarPostMix, self).__init__()

    def call(self, inputs):
        return tf.keras.layers.concatenate([self.mean(inputs), self.std(inputs)])

# %%
class MeanStdVarPriorMix(tf.keras.layers.Layer):
    def __init__(self, cat_prob, idx, batch_size=1):
        super(MeanStdVarPriorMix, self).__init__()
        self.mean = tf.keras.Sequential([
            tfp.layers.DenseVariational(4, posterior_mean_field, DistFromMix(cat_prob, idx, 0).fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(2, posterior_mean_field, DistFromMix(cat_prob, idx, 1).fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(1, posterior_mean_field, DistFromMix(cat_prob, idx, 2).fn, kl_weight=1/batch_size)
        ])
        self.std = tf.keras.Sequential([
            tfp.layers.DenseVariational(1, posterior_mean_field, DistFromMix(cat_prob, idx, 3).fn, kl_weight=1/batch_size)
        ])
    
    def build(self, input_shape):
        super(MeanStdVarPriorMix, self).__init__()

    def call(self, inputs):
        return tf.keras.layers.concatenate([self.mean(inputs), self.std(inputs)])
# %%
model = tf.keras.Sequential([
        MeanStdVarPostMix(w_mse_test, idx, prior_loc=0, prior_scale=2, batch_size=300),
        tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-9 + tf.math.softplus(1 * t[...,1:]))),
    ])
# dist = model(X_test_ext[:,np.newaxis]/X_MAX)
# %%
# num_random_draw = 1000
# yhats = dist.sample(num_random_draw).numpy()
# yhats = (yhats+1)*Y_MAX

num_random_draw = 100
yhats_dist = [model(X_test_ext[:,np.newaxis]/X_MAX) for _ in range(num_random_draw)]
yhats = np.vstack([dist.sample(num_random_draw).numpy() for dist in yhats_dist])
yhats = (yhats+1)*Y_MAX
# %%
Y_prcntl_prior = np.percentile(yhats[...,0], [2.5, 50.0, 97.5], axis=0)

fig = plt.figure()
plt.plot(X_test_ext, yhats[...,0].T, '.', color='C2', alpha=0.2)
plt.plot(X_test, Y_test, 'o', color='gray')
plt.plot(X_test[:num_test_points], Y_test[:num_test_points], 'ok')

plt.plot(X_test_ext, Y_prcntl_mse_test[1], '-', color='C1', linewidth=1., label='MSE')
plt.plot(X_test_ext, Y_prcntl_mse_test[0], '--', color='C1', linewidth=1.)
plt.plot(X_test_ext, Y_prcntl_mse_test[2], '--', color='C1', linewidth=1.)

plt.plot(X_test_ext, Y_prcntl_prior[1], '-', color='C2', linewidth=1., label='Prior')
plt.plot(X_test_ext, Y_prcntl_prior[0], '--', color='C2', linewidth=1.)
plt.plot(X_test_ext, Y_prcntl_prior[2], '--', color='C2', linewidth=1.)
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()


# %%
class MeanStdVarPrior(tf.keras.layers.Layer):
    def __init__(self, prior_loc=np.zeros(4), prior_scale=np.ones(4)*2, batch_size=1):
        super(MeanStdVarPrior, self).__init__()
        self.mean = tf.keras.Sequential([
            tfp.layers.DenseVariational(4, posterior_mean_field, PriorDist(prior_loc[0], prior_scale[0]).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(2, posterior_mean_field, PriorDist(prior_loc[1], prior_scale[1]).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(1, posterior_mean_field, PriorDist(prior_loc[2], prior_scale[2]).prior_fn, kl_weight=1/batch_size)
        ])
        self.std = tf.keras.Sequential([
            tfp.layers.DenseVariational(1, posterior_mean_field, PriorDist(prior_loc[3], prior_scale[3]).prior_fn, kl_weight=1/batch_size)
        ])
    
    def build(self, input_shape):
        super(MeanStdVarPrior, self).__init__()

    def call(self, inputs):
        return tf.keras.layers.concatenate([self.mean(inputs), self.std(inputs)])
# %%
X = X_test[:num_test_points]
X_norm = X / X_MAX
Y = Y_test[:num_test_points]
Y_norm = Y / Y_MAX - 1

loc = []
scale = []
for i in range(len(prior_weights)):
    l,s = np.split(prior_weights[i],2)
    loc.append(l)
    scale.append(1e-5 + tf.math.softplus(s).numpy())

model = tf.keras.Sequential([
        MeanStdVarPrior(prior_loc=loc, prior_scale=scale, batch_size=100000),
        tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-9 + tf.math.softplus(1 * t[...,1:]))),
    ])

# model.build(input_shape=(X.shape[0], 1))
# model.set_weights(model_dic_list[np.argmax(w_mse_full_test)]['weights'])

# fit
history = train_model(model, X_norm, Y_norm, 100+num_test_points, batch_size=X_norm.shape[0], epochs=10000).history

# %%

num_random_draw = 100
yhats_dist = [model(X_test_ext[:,np.newaxis]/X_MAX) for _ in range(num_random_draw)]
yhats = np.vstack([dist.sample(num_random_draw).numpy() for dist in yhats_dist])
yhats = (yhats+1)*Y_MAX
# %%
Y_prcntl_prior = np.percentile(yhats[...,0], [2.5, 50.0, 97.5], axis=0)

fig = plt.figure()
# plt.plot(X_test_ext, yhats[...,0].T, '.', color='C2', alpha=0.2)
plt.plot(X_test, Y_test, 'o', color='gray')
plt.plot(X_test[:num_test_points], Y_test[:num_test_points], 'ok')

plt.plot(X_test_ext, Y_prcntl_mse_test[1], '-', color='C1', linewidth=1., label='MSE')
plt.plot(X_test_ext, Y_prcntl_mse_test[0], '--', color='C1', linewidth=1.)
plt.plot(X_test_ext, Y_prcntl_mse_test[2], '--', color='C1', linewidth=1.)

plt.plot(X_test_ext, Y_prcntl_prior[1], '-', color='C2', linewidth=1., label='Prior')
plt.plot(X_test_ext, Y_prcntl_prior[0], '--', color='C2', linewidth=1.)
plt.plot(X_test_ext, Y_prcntl_prior[2], '--', color='C2', linewidth=1.)
plt.xlabel('Cumulative Energy (kWh)')
plt.ylabel(r'$q_{MAX}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()


# %%



plt.show()

# %%

if args.save:
    # clean tf keras models from list before saving
    for batt_i in range(n_batt):
        del model_dic_list[batt_i]['model']

    np.save(SAVE_DATA_PATH, model_dic_list)