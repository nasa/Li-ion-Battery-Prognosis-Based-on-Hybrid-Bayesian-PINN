# imports
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

matplotlib.rc('font', size=14)

from aging_model_train import get_model
from aging_model_data import *

PLOT_Y_LEGEND = r'$q_{MAX}$'
SAVING_STR = '_batt_{}_q_max'.format(skip[0]+1)
Y_ref = q_max_ref
Y_LIM = Y_LIM_q_max
Y_MAX = Y_MAX_q_max
Y_test = Y_test_q_max

model_dic_list = model_dic_list_q_max

def get_mu_s_from_model(model,X,num_random_draw=100):
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

def get_ensemble_w_mse(model_dic_list, X_test, Y_test):
    mse_test = np.zeros(n_batt)
    mean_models = np.zeros((X_test.shape[0],n_batt))
    std_models = np.zeros((X_test.shape[0],n_batt))

    for batt_i in range(n_batt):
        m_avg, s_avg = get_mu_s_from_model(model_dic_list[batt_i]['model'], X_test)

        m = (m_avg+1)*Y_MAX
        lb = ((m_avg - 2*s_avg) +1)*Y_MAX
        ub = ((m_avg + 2*s_avg) +1)*Y_MAX

        s = (ub-lb)/4

        mean_models[:,batt_i]  = m
        std_models[:,batt_i]  = s

        mse_test[batt_i] = np.mean((m - Y_test)**2)

    Cinv = np.linalg.inv(np.diag(mse_test[idx]))
    e = np.ones((n_batt-1,1))
    lamb = 2/(e.T@Cinv@e)
    w_mse = (lamb * (Cinv@e)/2)
    w_mse_full = np.zeros(n_batt)
    w_mse_full[idx] = w_mse[:,0]

    return w_mse_full, mean_models, std_models

def get_ensemble_w_like(X_test, Y_test):
    like = np.zeros((X_test.shape[0],n_batt))
    mean_models = np.zeros((X_test.shape[0],n_batt))
    std_models = np.zeros((X_test.shape[0],n_batt))

    for batt_i in range(n_batt):
        m_avg, s_avg = get_mu_s_from_model(model_dic_list[batt_i]['model'], X_test)

        m = (m_avg+1)*Y_MAX
        lb = ((m_avg - 2*s_avg) +1)*Y_MAX
        ub = ((m_avg + 2*s_avg) +1)*Y_MAX

        s = (ub-lb)/4

        mean_models[:,batt_i]  = m
        std_models[:,batt_i]  = s

        mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
        like[:,batt_i] = mvn.log_prob(Y_test)

    w = np.sum(like[:,idx], axis=0)
    w_norm = (w-w.min())/(w.max()-w.min())
    w_norm = w_norm / np.sum(w_norm)

    w_norm_full = np.zeros(n_batt)
    w_norm_full[idx] = w_norm

    return w_norm_full, mean_models, std_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=False, action="store_true" , help="Save results")
    parser.add_argument("--r0", default=False, action="store_true" , help="Train for R_0 [q_max otherwhise]")
    args = parser.parse_args()

    if args.r0:
        PLOT_Y_LEGEND = r'$R_0$'
        SAVING_STR = SAVING_STR = '_batt_{}_R_0'.format(skip[0]+1)
        Y_ref = R_0_ref
        Y_LIM = Y_LIM_R_0
        Y_MAX = Y_MAX_R_0
        Y_test = Y_test_R_0

        model_dic_list = model_dic_list_R_0
        
    fig = plt.figure()
    plt.plot(X_test, Y_test, '.', color='black')

    for batt_i in range(n_batt):
        if batt_i in skip:
            continue

        X = cum_kWh_ref[batt_i]
        Y = Y_ref[batt_i]

        m = (model_dic_list[batt_i]['avg_m']+1)*Y_MAX
        lb = ((model_dic_list[batt_i]['avg_m'] - 2*model_dic_list[batt_i]['avg_s']) +1)*Y_MAX
        ub = ((model_dic_list[batt_i]['avg_m'] + 2*model_dic_list[batt_i]['avg_s']) +1)*Y_MAX

        plt.fill_between(X, ub, lb, alpha=0.2, color='C{}'.format(batt_i))
        plt.plot(X, Y, '.', color='C{}'.format(batt_i))
        plt.plot(X, m, '-', color='C{}'.format(batt_i))

    plt.grid()
    plt.xlabel('Cumulative Energy (kWh)')
    plt.ylabel(PLOT_Y_LEGEND)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # %% Likelihood for test data in all models

    w_norm_full, mean_models, std_models = get_ensemble_w_like(X_test, Y_test)

    fig = plt.figure()
    plt.plot(X_test, Y_test, '.', color='black')

    for batt_i in range(n_batt):
        if batt_i in skip:
            continue

        m = mean_models[:,batt_i]
        s = std_models[:,batt_i]

        lb = m - 2*s
        ub = m + 2*s

        plt.fill_between(X_test, ub, lb, alpha=0.2, color='C{}'.format(batt_i))
        plt.plot(X_test, m, '-', color='C{}'.format(batt_i))

    plt.grid()
    plt.xlabel('Cumulative Energy (kWh)')
    plt.ylabel(PLOT_Y_LEGEND)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # %% ensemble weights from likelihood

    n_samples = np.round(1000*w_norm_full)
    Y_samples = []

    for batt_i in range(n_batt):
        if batt_i in skip:
            continue
        
        m = mean_models[:,batt_i]
        s = std_models[:,batt_i]

        mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
        Y_samples.append(mvn.sample(n_samples[batt_i]))

    Y_samples = np.vstack(Y_samples).T

    Y_prcntl = np.percentile(Y_samples, [2.5, 50.0, 97.5], axis=1)

    # fig = plt.figure()
    # plt.plot(X_test, Y_test, 'ok')
    # plt.plot(X_test, Y_prcntl[1], '-b', linewidth=1.)
    # plt.plot(X_test, Y_prcntl[0], '--b', linewidth=1.)
    # plt.plot(X_test, Y_prcntl[2], '--b', linewidth=1.)
    # plt.xlabel('Cumulative Energy (kWh)')
    # plt.ylabel(PLOT_Y_LEGEND)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # plt.grid()

    # %% ensemble weights from MSE
    w_mse_full, mean_models, std_models = get_ensemble_w_mse(model_dic_list, X_test, Y_test)

    n_samples_mse = np.round(1000*w_mse_full)
    Y_samples_mse = []

    for batt_i in range(n_batt):
        if batt_i in skip:
            continue
        
        m = mean_models[:,batt_i]
        s = std_models[:,batt_i]

        mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
        Y_samples_mse.append(mvn.sample(n_samples_mse[batt_i]))

    Y_samples_mse = np.vstack(Y_samples_mse).T
    Y_prcntl_mse = np.percentile(Y_samples_mse, [2.5, 50.0, 97.5], axis=1)

    # Getting samples from mixture
    mix = tfd.Mixture(
        cat=tfd.Categorical(probs=w_mse_full[idx]),
        components=[
            tfd.MultivariateNormalDiag(loc=mean_models[:,batt_i], scale_diag=std_models[:,batt_i])
            for batt_i in idx
            ]
    )
    Y_samples_mix = mix.sample(1000).numpy()
    Y_prcntl_mix = np.percentile(Y_samples_mix, [2.5, 50.0, 97.5], axis=0)


    fig = plt.figure()
    plt.plot(X_test, Y_test, 'ok')

    plt.plot(X_test, Y_prcntl[1], '-', color='C0', label='Likelihood')
    plt.plot(X_test, Y_prcntl[0], '--', color='C0', linewidth=1.)
    plt.plot(X_test, Y_prcntl[2], '--', color='C0', linewidth=1.)

    plt.plot(X_test, Y_prcntl_mse[1], '-', color='C1', label='MSE')
    plt.plot(X_test, Y_prcntl_mse[0], '--', color='C1', linewidth=1.)
    plt.plot(X_test, Y_prcntl_mse[2], '--', color='C1', linewidth=1.)

    plt.plot(X_test, Y_prcntl_mix[1], '-', color='C2', label='MIX MSE')
    plt.plot(X_test, Y_prcntl_mix[0], '--', color='C2', linewidth=1.)
    plt.plot(X_test, Y_prcntl_mix[2], '--', color='C2', linewidth=1.)
    plt.xlabel('Cumulative Energy (kWh)')
    plt.ylabel(PLOT_Y_LEGEND)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.grid()
    plt.legend()

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

    plt.show()

    if not args.save:
        SAVE = input("Save results (Y/N): ").lower()=='y'
    else:
        SAVE = True
    
    if SAVE:
        fig.savefig('./figures/ensemble_weights{}.png'.format(SAVING_STR))

        save_dic = {
            'ens_w_like': w_norm_full,
            'ens_w_mse': w_mse_full
        }

        np.save('./training/ensemble_weights{}.npy'.format(SAVING_STR), save_dic)

