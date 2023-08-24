import numpy as np
import math
from time import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
import tensorflow as tf

from model import get_model

from battery_data import getDischargeMultipleBatteries

matplotlib.rc('font', size=14)

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)
tf.get_logger().setLevel('ERROR')  # supress warnings and infos

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


SIMULATION_OVER_STEPS = 200
inputs_shiffed = np.hstack([inputs_shiffed, inputs_shiffed[:, -SIMULATION_OVER_STEPS:]])
inputs = np.hstack([inputs, inputs[:, -SIMULATION_OVER_STEPS:]])
time_window_size = inputs_shiffed.shape[1]

inputs_shiffed_all = inputs_shiffed
inputs_all = inputs
target_all = target
target_shiffed_all = target_shiffed
reach_EOD_all = reach_EOD

xi = np.linspace(0.0,1.0,100)

reach_EOD_sort_idx = np.argsort(reach_EOD)

K_FOLDS = 6
val_size = int(inputs.shape[0]//K_FOLDS)
train_size = inputs.shape[0]-val_size

mse_all = np.zeros((K_FOLDS, inputs.shape[0]))
q_max_all = np.zeros((K_FOLDS, inputs.shape[0]))
R_0_all = np.zeros((K_FOLDS, inputs.shape[0]))
mlp_all = np.zeros((K_FOLDS, 100))

val_idx_all = np.zeros((K_FOLDS, val_size), dtype=int)
train_idx_all = np.zeros((K_FOLDS, train_size), dtype=int)

without_CI_list = []

for kfold in range(K_FOLDS):
    print("")
    print(" * * Evaluating K-Fold {}/{} * * ".format(kfold+1,K_FOLDS))

    checkpoint_filepath = './training/cp_mlp_kfold_{}.ckpt'.format(kfold)

    val_idx = reach_EOD_sort_idx[kfold::K_FOLDS]
    train_idx = [i for i in np.arange(0,36) if i not in val_idx]

    val_idx_all[kfold, :] = val_idx
    train_idx_all[kfold, :] = train_idx

    inputs_shiffed = inputs_shiffed_all[train_idx,:,:]
    inputs = inputs_all[train_idx,:,:]
    target = target_all[train_idx,:]
    target_shiffed = target_shiffed_all[train_idx,:]
    reach_EOD = reach_EOD_all[train_idx]

    model_eval = get_model(batch_input_shape=(1,time_window_size-SIMULATION_OVER_STEPS,1), dt=dt, mlp=True, share_q_r=False)
    model_eval.compile(optimizer='adam', loss="mse", metrics=["mae"])
    model = get_model(batch_input_shape=inputs.shape, dt=dt, mlp=True, share_q_r=False)
    model.compile(optimizer='adam', loss="mse", metrics=["mae"])

    model.load_weights(checkpoint_filepath)

    weights = model.get_weights()

    pred_shiffed = model.predict(inputs_shiffed)[:,:,0]

    pred = np.full((inputs.shape[0],inputs.shape[1]), np.nan)
    for i in range(pred.shape[0]):
        pred[i, :(reach_EOD[i]+SIMULATION_OVER_STEPS)] = pred_shiffed[i, (max_size - reach_EOD[i]):]


    mse = np.zeros(inputs.shape[0])
    weights_eval = weights.copy()
    for i in range(inputs.shape[0]):
        weights_eval[0] = np.reshape(weights[0][i], (1,))
        weights_eval[1] = np.reshape(weights[1][i], (1,))
        model_eval.set_weights(weights_eval)
        mse[i] = model_eval.evaluate(inputs_shiffed[i,:target_shiffed.shape[1],:][np.newaxis,:,:], target_shiffed[i,:][np.newaxis,:,np.newaxis], verbose=0)[0]
        # print("MSE[{}]: {}".format(i, mse[i]))

    print("AVG MSE:, ", mse.mean())

    mse_all[kfold, train_idx] = mse
    q_max_all[kfold, train_idx] = weights[0]*model.layers[0].cell.qMaxBASE.numpy()
    R_0_all[kfold, train_idx] = weights[1]*model.layers[0].cell.RoBASE.numpy()
    mlp_all[kfold, :] = model.layers[0].cell.MLPp(xi[:,np.newaxis])[:,0]

    pred_lb = []
    pred_ub = []
    for i in range(pred.shape[1]):
        up = np.percentile(pred[:,i], 92.5)
        lb = np.percentile(pred[:,i], 7.5)
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
    
    without_CI = (total_samples_pts - within_CI) / total_samples_pts
    plt.plot(i*np.ones(np.sum(~within)), target_all[val_idx,i][~within], '+k', markersize=3, label='Pts out CI - {:.1f}%'.format(without_CI*100))

    plt.ylim([3.2,4.2])
    plt.grid()
    plt.legend()

    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    plt.title('K-fold {}/{}'.format(kfold+1,K_FOLDS))
    fig.savefig('./figures/k_fold_CI_pts_fold_{}.png'.format(kfold+1))

    without_CI_list.append(without_CI)
    
print("")
print("* * * *")
print("Pts out CI Mean:", np.mean(without_CI_list))
print("Pts out CI Std:", np.std(without_CI_list))
print("* * * *")
print("")

fig = plt.figure()
plt.hist(without_CI_list)
plt.xlabel('% Pts out CI - Mean: {:.2f} Std: {:.2f}'.format(np.mean(without_CI_list)*100, np.std(without_CI_list)*100))
plt.grid()
fig.savefig('./figures/k_fold_CI_dist.png')

fig = plt.figure()
for i in range(K_FOLDS):
    plt.bar([int(i+1)], [mse_all[i,:].mean()])
plt.ylabel(r'mse')
plt.xlabel('K-Folds')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()
fig.savefig('./figures/k_fold_MSE.png')

# load model weights of base train
checkpoint_filepath_base = './training/cp_mlp_save4.ckpt'
model_base = get_model(batch_input_shape=inputs_all.shape, dt=dt, mlp=True, share_q_r=False)
model_base.load_weights(checkpoint_filepath_base)
weights_base = model_base.get_weights()

mse_base = np.zeros(inputs_all.shape[0])
weights_eval = weights_base.copy()
for i in range(inputs_all.shape[0]):
    weights_eval[0] = np.reshape(weights_base[0][i], (1,))
    weights_eval[1] = np.reshape(weights_base[1][i], (1,))
    model_eval.set_weights(weights_eval)
    mse_base[i] = model_eval.evaluate(inputs_shiffed_all[i,:target_shiffed_all.shape[1],:][np.newaxis,:,:], target_shiffed_all[i,:][np.newaxis,:,np.newaxis], verbose=0)[0]
    # print("MSE[{}]: {}".format(i, mse[i]))

fig = plt.figure()
# sns.kdeplot(mse_base, shade=True, color='gray')
# for i in range(mse_all.shape[0]):
#     # plt.hist(mse_all[i,train_idx_all[i]], alpha=0.2)
#     sns.kdeplot(mse_all[i,train_idx_all[i]], linestyle='--')
# plt.xlabel(r'mse')
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# plt.grid()
X_plot = np.linspace(0.0, 5e-4)
kde = KernelDensity(kernel='gaussian', bandwidth=3e-5).fit(mse_base[:,np.newaxis])
log_dens = kde.score_samples(X_plot[:,np.newaxis])
plt.fill_between(X_plot, np.exp(log_dens), color='gray', alpha=0.3)
for i in range(R_0_all.shape[0]):
    kde = KernelDensity(kernel='gaussian', bandwidth=3e-5).fit(mse_all[i,train_idx_all[i]][:,np.newaxis])
    log_dens = kde.score_samples(X_plot[:,np.newaxis])
    plt.plot(X_plot, np.exp(log_dens), '--')
plt.xlabel(r'mse')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.grid()
plt.xlim([0,5e-4])
plt.ylim([0,12000])
fig.savefig('./figures/k_fold_MSE_dist.png')

fig = plt.figure()
sns.kdeplot(weights[0]*model_base.layers[0].cell.qMaxBASE.numpy(), shade=True, color='gray')
for i in range(q_max_all.shape[0]):
    # plt.hist(q_max_all[i,train_idx_all[i]], alpha=0.2)
    sns.kdeplot(q_max_all[i,train_idx_all[i]], linestyle='--')
plt.xlabel(r'$q_{max}$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.grid()
fig.savefig('./figures/k_fold_q_max_dist.png')

fig = plt.figure()
# sns.kdeplot(weights_base[1]*model_base.layers[0].cell.RoBASE.numpy(), shade=True, color='gray')
# for i in range(R_0_all.shape[0]):
#     # plt.hist(R_0_all[i,train_idx_all[i]], alpha=0.2)
#     sns.kdeplot(R_0_all[i,train_idx_all[i]], linestyle='--')
# plt.xlabel(r'$R_0$')
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# plt.grid()
X_plot = np.linspace(0.0, 1.25e-1)
kde = KernelDensity(kernel='gaussian', bandwidth=1e-2).fit((weights_base[1]*model_base.layers[0].cell.RoBASE.numpy())[:,np.newaxis])
log_dens = kde.score_samples(X_plot[:,np.newaxis])
plt.fill_between(X_plot, np.exp(log_dens), color='gray', alpha=0.3)
for i in range(R_0_all.shape[0]):
    kde = KernelDensity(kernel='gaussian', bandwidth=1e-2).fit(R_0_all[i,train_idx_all[i]][:,np.newaxis])
    log_dens = kde.score_samples(X_plot[:,np.newaxis])
    plt.plot(X_plot, np.exp(log_dens), '--')
plt.xlabel(r'$R_0$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.grid()
plt.xlim([0,1.2e-1])
plt.ylim([0,30])
fig.savefig('./figures/k_fold_R_0_dist.png')


# time_axis = np.arange(time_window_size) * dt


xi = np.linspace(0.0,1.0,100)
fig = plt.figure('MLP')
plt.fill_between(xi, model_base.layers[0].cell.MLPp(xi[:,np.newaxis])[:,0], np.min(model_base.layers[0].cell.MLPp(xi[:,np.newaxis])[:,0]), color='gray', alpha=0.3)
plt.plot(xi, mlp_all.T, linestyle='--')
plt.xlabel(r'$x_p$')
plt.ylabel(r'$V_{INT}$')
plt.grid()
plt.tight_layout()
fig.savefig('./figures/k_fold_MLP_dist.png')


fig = plt.figure()
plt.boxplot(q_max_all)
plt.grid()
plt.ylim([np.min(q_max_all[q_max_all>0]), np.max(q_max_all[q_max_all>0])])
plt.ylabel(r'$q_{max}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
fig.savefig('./figures/k_fold_boxplot.png')

plt.show()
