# %%
# imports
import sys
import argparse
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

matplotlib.rc('font', size=14)

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

# sys.argv = ['']

parser = argparse.ArgumentParser()
parser.add_argument("--save", default=False, action="store_true" , help="Save results")
args = parser.parse_args()

from aging_model_data import *
from aging_model_train import get_model, train_model, PriorDist, posterior_mean_field, MeanStdVar
from aging_model_ensemble import get_ensemble_w_mse, get_mu_s_from_model
from aging_model_ens_test import MeanStdVarPrior, MeanStdVarPriorMix, MeanStdVarPostMix, DistFromMix
from BatteryRNNCell_mlp import BatteryRNNCell

# %% Model OBS only
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

df_columns=['metric',] + ['p_{:.1f}'.format(p*100) for p in quantiles] + ['obs_pts', 'obs_until', 'forecast_at', 'model_type']
df = pd.DataFrame(columns=df_columns)

dt = 10.0
# inputs = np.ones((1000,800,1))
dist_size = 1000
inputs = np.ones((dist_size*100,800,1))

# batch_size = inputs.shape[0]
batch_size = dist_size
batt_miss_idx = -1

bi = 0
# num_test_points = 26
NUM_PTS_LIST = [6,16,26,38]  # for batt 4
# NUM_PTS_LIST = [6,16,26,38,52]  # for batt 2
# NUM_PTS_LIST = [6]

# %%

stats_rmse = []
stats_eod = []

for num_test_points in NUM_PTS_LIST:
    print('')
    print('* * * * * * * ')
    print('Testing Battery #{} with {} obs pts. '.format(skip[0]+1, num_test_points), NUM_PTS_LIST)
    print('* * * * * * * ')

    print('Loading Models and Infering...')

    q_max_model = tf.keras.Sequential([
        MeanStdVar(),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-9 + tf.math.softplus(1.0 * t[...,1:])),
                            convert_to_tensor_fn=lambda s: s.sample(batch_size)),
        tf.keras.layers.Lambda(lambda x: (x+1)*Y_MAX_q_max)
    ])
    q_max_model.build(input_shape=(1,1))
    SAVING_STR = '_batt_{}_q_max'.format(skip[0]+1)
    q_max_model.set_weights(np.load('./training/test_solely_model_{}pts_weights{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True))


    R_0_model = tf.keras.Sequential([
        MeanStdVar(),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-9 + tf.math.softplus(1.0 * t[...,1:])),
                            convert_to_tensor_fn=lambda s: s.sample(batch_size)),
        tf.keras.layers.Lambda(lambda x: (x+1)*Y_MAX_R_0)
    ])
    R_0_model.build(input_shape=(1,1))
    SAVING_STR = '_batt_{}_R_0'.format(skip[0]+1)
    R_0_model.set_weights(np.load('./training/test_solely_model_{}pts_weights{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True))


    curr_cum_pwh = cum_kWh_ref[skip[0]][batt_miss_idx]/X_MAX

    cell = BatteryRNNCell(q_max_model, R_0_model, curr_cum_pwh=curr_cum_pwh, dtype=DTYPE, dt=dt, batch_size=inputs.shape[0])
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, batch_input_shape=inputs.shape, return_state=False, dtype=DTYPE)

    cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))

    output = rnn(inputs)[:,:,0].numpy()

    # output_list = []

    # for _ in range(100):
    #     output_list.append(rnn(inputs)[:,:,0].numpy())
    #     cell = BatteryRNNCell(q_max_model, R_0_model, curr_cum_pwh=curr_cum_pwh, dtype=DTYPE, dt=dt, batch_size=inputs.shape[0])
    #     rnn = tf.keras.layers.RNN(cell, return_sequences=True, batch_input_shape=inputs.shape, return_state=False, dtype=DTYPE)
    #     cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))

    # output = np.vstack(output_list)


    # %% Model POST ensemble

    # q_max
    SAVING_STR = '_batt_{}_q_max'.format(skip[0]+1)
    model_dic_list_post_q_max = np.load('./training/ensemble_model_dic_list_post_{}pts{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True)

    for batt_i in range(n_batt):
        X = cum_kWh_ref[batt_i]

        model = tf.keras.Sequential([
                MeanStdVarPrior(batch_size=300),
                tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :1],
                                    scale=1e-9 + tf.math.softplus(1 * t[...,1:]))),
            ])
        model.build(input_shape=(1,1))
        model.set_weights(model_dic_list_post_q_max[batt_i]['weights'])
        model_dic_list_post_q_max[batt_i]['model'] = model
        
    w_mse_full_q_max = np.load('./training/ensemble_weights_test_{}pts_prior{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True)
    w_mse_full_post_q_max = np.load('./training/ensemble_weights_test_{}pts_post{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True)


    q_max_model = tf.keras.Sequential([
        MeanStdVarPostMix(model_dic_list_post_q_max, w_mse_full_post_q_max[idx], idx, prior_loc=0, prior_scale=2, batch_size=300),
        tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-9 + tf.math.softplus(1 * t[...,1:])),
                            convert_to_tensor_fn=lambda s: s.sample(batch_size)),
        tf.keras.layers.Lambda(lambda x: (x+1)*Y_MAX_q_max),
    ])

    # _R_0
    SAVING_STR = '_batt_{}_R_0'.format(skip[0]+1)
    model_dic_list_post_R_0 = np.load('./training/ensemble_model_dic_list_post_{}pts{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True)

    for batt_i in range(n_batt):
        X = cum_kWh_ref[batt_i]

        model = tf.keras.Sequential([
                MeanStdVarPrior(batch_size=300),
                tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :1],
                                    scale=1e-9 + tf.math.softplus(1 * t[...,1:]))),
            ])
        model.build(input_shape=(1,1))
        model.set_weights(model_dic_list_post_R_0[batt_i]['weights'])
        model_dic_list_post_R_0[batt_i]['model'] = model
        
    w_mse_full_R_0 = np.load('./training/ensemble_weights_test_{}pts_prior{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True)
    w_mse_full_post_R_0 = np.load('./training/ensemble_weights_test_{}pts_post{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True)


    R_0_model = tf.keras.Sequential([
        MeanStdVarPostMix(model_dic_list_post_R_0, w_mse_full_post_R_0[idx], idx, prior_loc=0, prior_scale=2, batch_size=300),
        tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1],
                            scale=1e-9 + tf.math.softplus(1 * t[...,1:])),
                            convert_to_tensor_fn=lambda s: s.sample(batch_size)),
        tf.keras.layers.Lambda(lambda x: (x+1)*Y_MAX_R_0),
    ])

    curr_cum_pwh = cum_kWh_ref[skip[0]][batt_miss_idx]/X_MAX

    cell = BatteryRNNCell(q_max_model, R_0_model, curr_cum_pwh=curr_cum_pwh, dtype=DTYPE, dt=dt, batch_size=inputs.shape[0])
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, batch_input_shape=inputs.shape, return_state=False, dtype=DTYPE)

    cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))

    output_post = rnn(inputs)[:,:,0].numpy()

    # output_list = []

    # for _ in range(100):
    #     output_list.append(rnn(inputs)[:,:,0].numpy())
    #     cell = BatteryRNNCell(q_max_model, R_0_model, curr_cum_pwh=curr_cum_pwh, dtype=DTYPE, dt=dt, batch_size=inputs.shape[0])
    #     rnn = tf.keras.layers.RNN(cell, return_sequences=True, batch_input_shape=inputs.shape, return_state=False, dtype=DTYPE)
    #     cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))

    # output_post = np.vstack(output_list)

    # %%

    mean = np.quantile(output, 0.5, 0)
    lb = np.quantile(output, 0.025, 0)
    ub = np.quantile(output, 0.975, 0)
    std = output.std(axis=0)

    mean_post = np.quantile(output_post, 0.5, 0)
    lb_post = np.quantile(output_post, 0.025, 0)
    ub_post = np.quantile(output_post, 0.975, 0)
    std_post = output_post.std(axis=0)

    EOD = 3.2

    fig = plt.figure()

    plt_X = np.arange(inputs.shape[1])*10

    p0 = plt.plot(plt_X[:len(target_ref[skip[0]][batt_miss_idx])], target_ref[skip[0]][batt_miss_idx], '-k')

    plt.fill_between(plt_X[ub>=EOD], ub[ub>=EOD], lb[ub>=EOD], color='C0', alpha=0.2)
    p1 = plt.plot(plt_X[mean>=EOD], mean[mean>=EOD], '-', color='C0', label='Observations only')
    sub = len(plt_X[mean>=EOD])//10
    p2 = plt.plot(plt_X[mean>=EOD][::sub], mean[mean>=EOD][::sub], '+', color='C0', label='Observations only')
    p2_2 = plt.fill([],[], color='C0', alpha=0.2)

    plt.fill_between(plt_X[ub_post>=EOD], ub_post[ub_post>=EOD], lb_post[ub_post>=EOD], color='C2', alpha=0.2)
    p3 = plt.plot(plt_X[mean_post>=EOD], mean_post[mean_post>=EOD], '-', color='C2', label='Observations + fleet prior')
    p4 = plt.plot(plt_X[mean_post>=EOD][::sub], mean_post[mean_post>=EOD][::sub], '.', color='C2')
    p5 = plt.fill([],[], color='C2', alpha=0.2)

    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.grid()
    plt.legend([(p0[0]), (p1[0], p2[0], p2_2[0]), (p3[0], p4[0], p5[0])], ['Measurement', 'Observations only', 'Observations + fleet prior'], loc='upper right')
    plt.ylim([EOD,4.2])

    if args.save:
        fig.savefig('./figures/forecast_{}pts_batt_{}.png'.format(num_test_points, skip[0]+1))

    target_size = np.argmax(np.isnan(target_ref[skip[0]][batt_miss_idx]))
    # mse = tf.keras.losses.MSE(target_ref[skip[0]][batt_miss_idx][:target_size], output[:,:target_size]).numpy()
    # rmse = np.sqrt(mse)
    # mae = tf.keras.losses.MAE(target_ref[skip[0]][batt_miss_idx][:target_size], output[:,:target_size]).numpy()
    # error_eod = plt_X[target_size-1] - plt_X[np.argmin(output>=EOD, axis=1)-1]

    # mse_post = tf.keras.losses.MSE(target_ref[skip[0]][batt_miss_idx][:target_size], output_post[:,:target_size]).numpy()
    # rmse_post = np.sqrt(mse_post)
    # mae_post = tf.keras.losses.MAE(target_ref[skip[0]][batt_miss_idx][:target_size], output_post[:,:target_size]).numpy()
    # error_eod_post = plt_X[target_size-1] - plt_X[np.argmin(output_post>=EOD, axis=1)-1]


    print('Generating stats...')

    mse = []
    mse_post = []
    mae = []
    mae_post = []
    error_eod = []
    error_eod_post = []

    for i in range(inputs.shape[0]):
        # o = np.quantile(output,q,0)
        o = output[i]
        o = o[o>=EOD]
        mse.append(tf.keras.losses.MSE(target_ref[skip[0]][batt_miss_idx][:min(len(o),target_size)], o[:target_size]).numpy())
        mae.append(tf.keras.losses.MAE(target_ref[skip[0]][batt_miss_idx][:min(len(o),target_size)], o[:target_size]).numpy())
        error_eod.append(plt_X[target_size-1] - plt_X[len(o)-1])

        # o = np.quantile(output_post,q,0)
        o = output_post[i]
        o = o[o>=EOD]
        mse_post.append(tf.keras.losses.MSE(target_ref[skip[0]][batt_miss_idx][:min(len(o),target_size)], o[:target_size]).numpy())
        mae_post.append(tf.keras.losses.MAE(target_ref[skip[0]][batt_miss_idx][:min(len(o),target_size)], o[:target_size]).numpy())
        error_eod_post.append(plt_X[target_size-1] - plt_X[len(o)-1])

    rmse = np.sqrt(mse)
    rmse_post = np.sqrt(mse_post)

    mse = np.quantile(mse,quantiles)
    rmse = np.quantile(rmse,quantiles)
    mae = np.quantile(mae,quantiles)
    error_eod = np.quantile(error_eod,quantiles)

    mse_post = np.quantile(mse_post,quantiles)
    rmse_post = np.quantile(rmse_post,quantiles)
    mae_post = np.quantile(mae_post,quantiles)
    error_eod_post = np.quantile(error_eod_post,quantiles)

    stats_rmse.append({
        "label": 'Obs',
        "whislo": rmse[0],
        "q1": rmse[1],
        "med": rmse[2],
        "q3": rmse[3],
        "whishi": rmse[4],
        "fliers": []
    })
    stats_rmse.append({
        "label": 'Obs + fleet',
        "whislo": rmse_post[0],
        "q1": rmse_post[1],
        "med": rmse_post[2],
        "q3": rmse_post[3],
        "whishi": rmse_post[4],
        "fliers": []
    })

    stats_eod.append({
        "label": 'Obs',
        "whislo": error_eod[0],
        "q1": error_eod[1],
        "med": error_eod[2],
        "q3": error_eod[3],
        "whishi": error_eod[4],
        "fliers": []
    })
    stats_eod.append({
        "label": 'Obs + fleet',
        "whislo": error_eod_post[0],
        "q1": error_eod_post[1],
        "med": error_eod_post[2],
        "q3": error_eod_post[3],
        "whishi": error_eod_post[4],
        "fliers": []
    })


    # fig = plt.figure('box_rmse')
    # plt.boxplot(rmse, positions=[bi], showfliers=False)

    # fig = plt.figure('error_eod')
    # plt.boxplot(error_eod/plt_X[target_size-1], positions=[bi], showfliers=False)
    # bi += 1

    # fig = plt.figure('box_rmse')
    # plt.boxplot(rmse_post, positions=[bi], showfliers=False)

    # fig = plt.figure('error_eod')
    # plt.boxplot(error_eod_post/plt_X[target_size-1], positions=[bi], showfliers=False)
    # bi += 1

    print(' * * ')
    print('Test - Battery #{}'.format(skip[0]+1))
    print('Observations until {:.1f} kWh ({:} pts)'.format(cum_kWh_ref[skip[0]][num_test_points-1], num_test_points))
    print('Forecast at {:.1f} kWh'.format(cum_kWh_ref[skip[0]][batt_miss_idx]))

    print('')
    print('Obs only:')
    print('percentiles', ['   {:.1f}%'.format(p*100) for p in quantiles])
    print('MSE:       ', mse)
    print('RMSE:      ', rmse)
    print('MAE:       ', mae)
    print('EOD error: ', error_eod)
    print('EOD err %: ', np.array(error_eod)/plt_X[target_size-1])

    print('')
    print('Obs + fleet prior:')
    print('percentiles', ['   {:.1f}%'.format(p*100) for p in quantiles])
    print('MSE:       ', mse_post)
    print('RMSE:      ', rmse_post)
    print('MAE:       ', mae_post)
    print('EOD error: ', error_eod_post)
    print('EOD err %: ', np.array(error_eod_post)/plt_X[target_size-1])
    print(' * * ')


    # ['metric', 'p_2.5', 'p_25.0', 'p_50.0', 'p_75.0', 'p_97.5', 'obs_pts', 'obs_until', 'forecast_at', 'model_type']
    base_list = [num_test_points, np.round(cum_kWh_ref[skip[0]][num_test_points-1],1), np.round(cum_kWh_ref[skip[0]][batt_miss_idx],1), 'solely']
    df = df.append(
        pd.DataFrame(
            [['MSE'] + list(mse) + base_list], columns=df_columns)
        , ignore_index=True
    )
    df = df.append(
        pd.DataFrame(
            [['RMSE'] + list(rmse) + base_list], columns=df_columns)
        , ignore_index=True
    )
    df = df.append(
        pd.DataFrame(
            [['MAE'] + list(mae) + base_list], columns=df_columns)
        , ignore_index=True
    )
    df = df.append(
        pd.DataFrame(
            [['EOD error'] + list(error_eod) + base_list], columns=df_columns)
        , ignore_index=True
    )
    df = df.append(
        pd.DataFrame(
            [['EOD err %']+list(np.array(error_eod)/plt_X[target_size-1])+base_list], columns=df_columns)
        , ignore_index=True
    )

    base_list = [num_test_points, np.round(cum_kWh_ref[skip[0]][num_test_points-1],1), np.round(cum_kWh_ref[skip[0]][batt_miss_idx],1), 'fleet_post']
    df = df.append(
        pd.DataFrame(
            [['MSE']+ list(mse_post) +base_list], columns=df_columns)
        , ignore_index=True
    )
    df = df.append(
        pd.DataFrame(
            [['RMSE']+ list(rmse_post) +base_list], columns=df_columns)
        , ignore_index=True
    )
    df = df.append(
        pd.DataFrame(
            [['MAE']+ list(mae_post) +base_list], columns=df_columns)
        , ignore_index=True
    )
    df = df.append(
        pd.DataFrame(
            [['EOD error']+ list(error_eod_post) +base_list], columns=df_columns)
        , ignore_index=True
    )
    df = df.append(
        pd.DataFrame(
            [['EOD err %']+list(np.array(error_eod_post)/plt_X[target_size-1])+base_list], columns=df_columns)
        , ignore_index=True
    )
    # %%


# %%
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
axes.bxp(stats_rmse)
plt.ylabel('RMSE')
plt.grid()

if args.save:
    fig.savefig('./figures/forecast_rmse_batt{}.png'.format(skip[0]+1))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
axes.bxp(stats_eod)
plt.ylabel('Relative EOD Error')
plt.grid()

if args.save:
    fig.savefig('./figures/forecast_error_eod_batt{}.png'.format(skip[0]+1))

if args.save:
    df.to_csv('forecast_metrics_test_batt_{}.csv'.format(skip[0]+1))

plt.show()
