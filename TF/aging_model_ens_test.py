# %%
# imports
from os import path
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

matplotlib.rc('font', size=14)

from aging_model_data import *
from aging_model_train import get_model, train_model, PriorDist, posterior_mean_field, MeanStdVar
from aging_model_ensemble import get_ensemble_w_mse, get_mu_s_from_model

PLOT_Y_LEGEND = r'$q_{MAX}$'
SAVING_STR = '_batt_{}_q_max'.format(skip[0]+1)
Y_ref = q_max_ref
Y_LIM = Y_LIM_q_max
Y_MAX = Y_MAX_q_max
Y_test = Y_test_q_max

model_dic_list = model_dic_list_q_max

# %%
class DistFromMix(tfd.MultivariateNormalDiag):
    # _data = {
    #     'mix': None,
    #     'cat_prob': None,
    #     'layers_size': None,
    #     'loc': [],
    #     'scale': [],
    #     'prob_draw': None,
    #     'draw': None
    # }
    def __init__(self, data_dic, model_dic_list, cat_prob, idx, layer_i, total_layers=4, **kwargs):
        self.n = len(model_dic_list[0]['weights'][layer_i])//2
        self.layer_i = layer_i

        self._data = data_dic

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
                cat=tfd.Categorical(probs=self._data['cat_prob']),
                components=[
                    tfd.MultivariateNormalDiag(loc=self._data['loc'][i], scale_diag=1e-5 + tf.nn.softplus(self._data['scale'][i]))
                    for i in range(len(idx))
                    ]
            )

            self._data['prob_draw'] = tf.Variable(self._data['mix'].mean(), trainable=False)
        
        i,j = self._data['layers_slices'][self.layer_i]
        # self.mix = tfd.Mixture(
        #         # cat=tfd.Categorical(probs=np.tile(self._data['cat_prob'],self.n).T),
        #         cat=tfd.Categorical(probs=self._data['cat_prob']),
        #         components=[
        #             tfd.MultivariateNormalDiag(loc=self._data['loc'][k][i:j], scale_diag=1e-5 + tf.nn.softplus(self._data['scale'][k][i:j]))
        #             for k in range(len(idx))
        #             ]
        #     )
        # self.mix_list = [tfd.MultivariateNormalDiag(
        #         loc=self._data['loc'][k][i:j], scale_diag=1e-5 + tf.nn.softplus(self._data['scale'][k][i:j])
        #     ) for k in range(len(idx)) ]
        
        super(DistFromMix, self).__init__(loc=self._data['mix'].mean()[i:j], scale_diag=self._data['mix'].stddev()[i:j] )

    def get_draw(self, dist):
        if self.layer_i==0:
            self._data['draw'] = self._data['mix'].sample()

        i,j = self._data['layers_slices'][self.layer_i]
        return self._data['draw'][i:j]

    # def log_prob(self, w):
    #     # if self.layer_i==0:
    #     self._data['prob_draw'].assign(self._data['mix'].mean())

    #     i,j = self._data['layers_slices'][self.layer_i]
    #     self._data['prob_draw'][i:j].assign(w)
    #     ret = self._data['mix'].log_prob(self._data['prob_draw'])
    #     # ret = tf.reduce_sum(tf.stack([dist.log_prob(w) for dist in self.mix_list]) * self._data['cat_prob'])
    #     # tf.print(ret.shape)
    #     return ret


    def fn(self, kernel_size, bias_size=0, dtype=None):
        dist_self = self
        
        return tf.keras.Sequential([
            tfp.layers.DistributionLambda(lambda t: dist_self,
            convert_to_tensor_fn=dist_self.get_draw
            )
        ])

class MeanStdVarPostMix(MeanStdVar):
    def __init__(self, model_dic_list, cat_prob, idx, prior_loc=0, prior_scale=2, batch_size=1):
        super(MeanStdVar, self).__init__()
        self.data_dic = {}
        self.mean = tf.keras.Sequential([
            tfp.layers.DenseVariational(4, DistFromMix(self.data_dic, model_dic_list, cat_prob, idx, 0).fn, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(2, DistFromMix(self.data_dic, model_dic_list, cat_prob, idx, 1).fn, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(1, DistFromMix(self.data_dic, model_dic_list, cat_prob, idx, 2).fn, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size)
        ])
        self.std = tf.keras.Sequential([
            tfp.layers.DenseVariational(1, DistFromMix(self.data_dic, model_dic_list, cat_prob, idx, 3).fn, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size)
        ])

class MeanStdVarPriorMix(MeanStdVar):
    def __init__(self, cat_prob, idx, batch_size=1):
        super(MeanStdVar, self).__init__()
        self.data_dic = {}
        self.mean = tf.keras.Sequential([
            tfp.layers.DenseVariational(4, posterior_mean_field, DistFromMix(self.data_dic, cat_prob, idx, 0).fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(2, posterior_mean_field, DistFromMix(self.data_dic, cat_prob, idx, 1).fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(1, posterior_mean_field, DistFromMix(self.data_dic, cat_prob, idx, 2).fn, kl_weight=1/batch_size)
        ])
        self.std = tf.keras.Sequential([
            tfp.layers.DenseVariational(1, posterior_mean_field, DistFromMix(self.data_dic, cat_prob, idx, 3).fn, kl_weight=1/batch_size)
        ])

class MeanStdVarPrior(MeanStdVar):
    def __init__(self, prior_loc=np.zeros(4), prior_scale=np.ones(4)*2, batch_size=1):
        super(MeanStdVar, self).__init__()
        self.mean = tf.keras.Sequential([
            tfp.layers.DenseVariational(4, posterior_mean_field, PriorDist(prior_loc[0], prior_scale[0]).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(2, posterior_mean_field, PriorDist(prior_loc[1], prior_scale[1]).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(1, posterior_mean_field, PriorDist(prior_loc[2], prior_scale[2]).prior_fn, kl_weight=1/batch_size)
        ])
        self.std = tf.keras.Sequential([
            tfp.layers.DenseVariational(1, posterior_mean_field, PriorDist(prior_loc[3], prior_scale[3]).prior_fn, kl_weight=1/batch_size)
        ])

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=False, action="store_true" , help="Save results")
    parser.add_argument("--train", default=False, action="store_true" , help="Force train even if saved results exist")
    parser.add_argument("--r0", default=False, action="store_true" , help="Train for R_0 [q_max otherwhise]")
    args = parser.parse_args()

    # NUM_PTS_LIST = [6,16,26,38]  # for batt 4
    NUM_PTS_LIST = [6,16,26,38,52]  # for batt 2

    if args.r0:
        PLOT_Y_LEGEND = r'$R_0$'
        SAVING_STR = '_batt_{}_R_0'.format(skip[0]+1)
        Y_ref = R_0_ref
        Y_LIM = Y_LIM_R_0
        Y_MAX = Y_MAX_R_0
        Y_test = Y_test_R_0

        #remove R_0 outlier
        Y_test[np.argmax(Y_test_R_0)] = (Y_test_R_0[np.argmax(Y_test_R_0)-1]+Y_test_R_0[np.argmax(Y_test_R_0)+1])/2

        model_dic_list = model_dic_list_R_0

    TRAIN = True
    if path.exists('./training/ensemble_model_dic_list_post_{}pts{}.npy'.format(NUM_PTS_LIST[0],SAVING_STR)) and not args.train:
        TRAIN = False

    saved_ens = np.load('./training/ensemble_weights{}.npy'.format(SAVING_STR), allow_pickle=True).item()

    w_mse_full = saved_ens['ens_w_mse']

    # NUM_PTS_LIST = [16,26,38,52]
    # NUM_PTS_LIST = [16]
    for num_test_points in NUM_PTS_LIST:

        X = X_test[:num_test_points]
        X_norm = X / X_MAX
        Y = Y_test[:num_test_points]
        Y_norm = Y / Y_MAX - 1

        model_dic_list_post = []

        if TRAIN:
            for batt_i in range(n_batt):

                model_dic = {
                    'batt_i': batt_i
                }

                prior_weights = model_dic_list[batt_i]['weights']

                loc = []
                scale = []
                for i in range(len(prior_weights)):
                    l,s = np.split(prior_weights[i],2)
                    loc.append(l)
                    scale.append(1e-5 + tf.math.softplus(s).numpy())

                model = tf.keras.Sequential([
                        MeanStdVarPrior(prior_loc=loc, prior_scale=scale, batch_size=300),
                        tfp.layers.DistributionLambda(
                        lambda t: tfd.Normal(loc=t[..., :1],
                                            scale=1e-9 + tf.math.softplus(1 * t[...,1:]))),
                    ])

                # use prior weights as initial posterior weights
                model.build(input_shape=(X_norm.shape[0],1))
                model.set_weights(prior_weights)

                # fit
                history = train_model(model, X_norm, Y_norm, 100+num_test_points, batch_size=X_norm.shape[0], epochs=4000).history

                model_dic['history'] = history
                model_dic['final_loss'] = np.min(model_dic['history']['loss'])

                model_dic['weights'] = model.get_weights()
                model_dic['model'] = model

                ## %%
                X_test_ext = np.linspace(0, 3.5, 100)

                num_random_draw = 100
                yhats_dist = [model_dic_list[batt_i]['model'](X_test_ext[:,np.newaxis]/X_MAX) for _ in range(num_random_draw)]
                yhats = np.vstack([dist.sample(num_random_draw).numpy() for dist in yhats_dist])
                yhats = (yhats+1)*Y_MAX

                Y_prcntl_prior = np.percentile(yhats[...,0], [2.5, 50.0, 97.5], axis=0)

                yhats_dist = [model(X_test_ext[:,np.newaxis]/X_MAX) for _ in range(num_random_draw)]
                yhats = np.vstack([dist.sample(num_random_draw).numpy() for dist in yhats_dist])
                yhats = (yhats+1)*Y_MAX

                Y_prcntl_post = np.percentile(yhats[...,0], [2.5, 50.0, 97.5], axis=0)
                ## %%

                model_dic['avg_m'] = np.mean(yhats[...,0], axis=0)
                model_dic['avg_s'] = np.std(yhats[...,0], axis=0)
                model_dic['prcntl'] = Y_prcntl_post

                model_dic_list_post.append(model_dic)

                fig = plt.figure()
                # plt.plot(X_test_ext, yhats[...,0].T, '.', color='C2', alpha=0.2)

                plt.plot(cum_kWh_ref[batt_i], Y_ref[batt_i], 'o', color='C1')

                plt.plot(X_test_ext, Y_prcntl_prior[1], '-', color='C1', linewidth=1., label='Prior')
                plt.plot(X_test_ext, Y_prcntl_prior[0], '--', color='C1', linewidth=1.)
                plt.plot(X_test_ext, Y_prcntl_prior[2], '--', color='C1', linewidth=1.)

                plt.plot(X_test, Y_test, 'o', color='gray')
                plt.plot(X_test[:num_test_points], Y_test[:num_test_points], 'ok')

                plt.plot(X_test_ext, Y_prcntl_post[1], '-', color='C2', linewidth=1., label='Post')
                plt.plot(X_test_ext, Y_prcntl_post[0], '--', color='C2', linewidth=1.)
                plt.plot(X_test_ext, Y_prcntl_post[2], '--', color='C2', linewidth=1.)
                plt.xlabel('Cumulative Energy (kWh)')
                plt.ylabel(PLOT_Y_LEGEND)
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                plt.grid()
                if args.save:
                    fig.savefig('./figures/post_{}pts_batt_{}{}.png'.format(num_test_points, batt_i+1, SAVING_STR))

            w_mse_full, mean_models, std_models = get_ensemble_w_mse(model_dic_list, X_test[:num_test_points], Y_test[:num_test_points])
            if args.save:
                np.save('./training/ensemble_weights_test_{}pts_prior{}.npy'.format(num_test_points, SAVING_STR), w_mse_full)

            w_mse_full_post, mean_models_post, std_models_post = get_ensemble_w_mse(model_dic_list_post, X_test[:num_test_points], Y_test[:num_test_points])
            if args.save:
                np.save('./training/ensemble_weights_test_{}pts_post{}.npy'.format(num_test_points, SAVING_STR), w_mse_full_post)

            # train solely model (without ensemble prior)
            solely_model = tf.keras.Sequential([
                    MeanStdVar(prior_loc=0, prior_scale=2, batch_size=300),
                    tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(loc=t[..., :1],
                                        scale=1e-9 + tf.math.softplus(1 * t[...,1:]))),
                ])

            history = train_model(solely_model, X_norm, Y_norm, 100+num_test_points, batch_size=X_norm.shape[0], epochs=4000).history

            if args.save:
                np.save('./training/test_solely_model_{}pts_weights{}.npy'.format(num_test_points, SAVING_STR), solely_model.get_weights())


        else:
            model_dic_list_post = np.load('./training/ensemble_model_dic_list_post_{}pts{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True)

            for batt_i in range(n_batt):
                X = cum_kWh_ref[batt_i]

                model = tf.keras.Sequential([
                        MeanStdVarPrior(batch_size=300),
                        tfp.layers.DistributionLambda(
                        lambda t: tfd.Normal(loc=t[..., :1],
                                            scale=1e-9 + tf.math.softplus(1 * t[...,1:]))),
                    ])
                model.build(input_shape=(X_norm.shape[0],1))
                model.set_weights(model_dic_list_post[batt_i]['weights'])
                model_dic_list_post[batt_i]['model'] = model
                
            w_mse_full = np.load('./training/ensemble_weights_test_{}pts_prior{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True)
            w_mse_full_post = np.load('./training/ensemble_weights_test_{}pts_post{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True)

            solely_model = tf.keras.Sequential([
                    MeanStdVar(prior_loc=0, prior_scale=2, batch_size=300),
                    tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(loc=t[..., :1],
                                        scale=1e-9 + tf.math.softplus(1 * t[...,1:]))),
                ])
            solely_model.build(input_shape=(X_norm.shape[0],1))
            solely_model.set_weights(np.load('./training/test_solely_model_{}pts_weights{}.npy'.format(num_test_points, SAVING_STR), allow_pickle=True))

        # PRIOR
        n_samples_mse_test = np.round(1000*w_mse_full)
        Y_samples_mse_test = []
        X_test_ext = np.linspace(0, 3.5, 100)

        for batt_i in range(n_batt):
            if batt_i in skip:
                continue

            m_avg, s_avg = get_mu_s_from_model(model_dic_list[batt_i]['model'], X_test_ext)

            m = (m_avg+1)*Y_MAX
            lb = ((m_avg - 2*s_avg) +1)*Y_MAX
            ub = ((m_avg + 2*s_avg) +1)*Y_MAX

            s = (ub-lb)/4

            mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
            Y_samples_mse_test.append(mvn.sample(n_samples_mse_test[batt_i]))

        Y_samples_mse_test = np.vstack(Y_samples_mse_test).T

        Y_prcntl_prior = np.percentile(Y_samples_mse_test, [2.5, 50.0, 97.5], axis=1)

        # POSTERIOR
        n_samples_mse_test = np.round(1000*w_mse_full_post)
        Y_samples_mse_test = []
        X_test_ext = np.linspace(0, 3.5, 100)

        for batt_i in range(n_batt):
            if batt_i in skip:
                continue

            m_avg, s_avg = get_mu_s_from_model(model_dic_list_post[batt_i]['model'], X_test_ext)

            m = (m_avg+1)*Y_MAX
            lb = ((m_avg - 2*s_avg) +1)*Y_MAX
            ub = ((m_avg + 2*s_avg) +1)*Y_MAX

            s = (ub-lb)/4

            mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
            Y_samples_mse_test.append(mvn.sample(n_samples_mse_test[batt_i]))

        Y_samples_mse_test = np.vstack(Y_samples_mse_test).T
        Y_prcntl_post = np.percentile(Y_samples_mse_test, [2.5, 50.0, 97.5], axis=1)

        # POSTERIOR use prior weights
        n_samples_mse_test = np.round(1000*w_mse_full)
        Y_samples_mse_test = []
        X_test_ext = np.linspace(0, 3.5, 100)

        for batt_i in range(n_batt):
            if batt_i in skip:
                continue

            m_avg, s_avg = get_mu_s_from_model(model_dic_list_post[batt_i]['model'], X_test_ext)

            m = (m_avg+1)*Y_MAX
            lb = ((m_avg - 2*s_avg) +1)*Y_MAX
            ub = ((m_avg + 2*s_avg) +1)*Y_MAX

            s = (ub-lb)/4

            mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
            Y_samples_mse_test.append(mvn.sample(n_samples_mse_test[batt_i]))

        Y_samples_mse_test = np.vstack(Y_samples_mse_test).T
        Y_prcntl_post2 = np.percentile(Y_samples_mse_test, [2.5, 50.0, 97.5], axis=1)

        # SOLELY model
        num_random_draw = 100
        yhats_dist = [solely_model(X_test_ext[:,np.newaxis]/X_MAX) for _ in range(num_random_draw)]
        yhats = np.vstack([dist.sample(num_random_draw).numpy() for dist in yhats_dist])
        yhats = (yhats+1)*Y_MAX

        Y_prcntl_solely = np.percentile(yhats[...,0], [2.5, 50.0, 97.5], axis=0)


        fig = plt.figure()
        line_x = X_test[num_test_points-1] + (X_test[num_test_points-2]-X_test[num_test_points-3])/2
        plt.plot([line_x,line_x], Y_LIM, '--', color='gray')
        plt.text(line_x+0.1, Y_LIM[1]-Y_LIM[1]*0.02, 'Forecast', {'color': 'gray', 'fontsize': 10, 'va': 'top'})
        plt.annotate("", xy=[line_x, Y_LIM[1]-Y_LIM[1]*0.06], xytext=[line_x+0.65, Y_LIM[1]-Y_LIM[1]*0.06],
                    arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", color='gray'))

        plt.plot(X_test, Y_test, 'ok', fillstyle='none')
        plt.plot(X_test[:num_test_points], Y_test[:num_test_points], 'ok')

        p1 = plt.plot(X_test_ext, Y_prcntl_prior[1], '-', color='C1', label='Fleet prior')
        sub = len(X_test_ext)//10
        p2 = plt.plot(X_test_ext[::sub], Y_prcntl_prior[1][::sub], 'x', color='C1', label='Fleet prior')
        plt.plot(X_test_ext, Y_prcntl_prior[0], '--', color='C1', linewidth=1.)
        plt.plot(X_test_ext, Y_prcntl_prior[2], '--', color='C1', linewidth=1.)

        plt.fill_between(X_test_ext, Y_prcntl_post[2], Y_prcntl_post[0], color='C2', alpha=0.2)
        p3 = plt.plot(X_test_ext, Y_prcntl_post[1], '-', color='C2', label='Observations + fleet prior')
        p4 = plt.plot(X_test_ext[::sub], Y_prcntl_post[1][::sub], '.', color='C2')
        p5 = plt.fill([],[], color='C2', alpha=0.2)

        plt.xlabel('Cumulative Energy (kWh)')
        plt.ylabel(PLOT_Y_LEGEND)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.grid()
        plt.legend([(p1[0], p2[0]), (p3[0], p4[0], p5[0])], ['Fleet prior', 'Observations + fleet prior'], loc='lower left')
        plt.xlim(X_LIM)
        plt.ylim(Y_LIM)
        if args.save:
            fig.savefig('./figures/ensemble_post_{}pts{}.png'.format(num_test_points, SAVING_STR))


        fig = plt.figure()
        plt.plot(X_test, Y_test, 'o', color='gray')
        plt.plot(X_test[:num_test_points], Y_test[:num_test_points], 'ok')

        plt.plot(X_test_ext, Y_prcntl_prior[1], '-.', color='C1', linewidth=1., label='Fleet prior')
        plt.plot(X_test_ext, Y_prcntl_prior[0], '--', color='C1', linewidth=1.)
        plt.plot(X_test_ext, Y_prcntl_prior[2], '--', color='C1', linewidth=1.)

        plt.fill_between(X_test_ext, Y_prcntl_post2[2], Y_prcntl_post2[0], color='C3', alpha=0.2)
        plt.plot(X_test_ext, Y_prcntl_post2[1], '-', color='C3', linewidth=1., label='Post w/ prior w')

        plt.xlabel('Cumulative Energy (kWh)')
        plt.ylabel(PLOT_Y_LEGEND)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.grid()
        plt.legend()
        if args.save:
            fig.savefig('./figures/ensemble_post2_{}pts{}.png'.format(num_test_points, SAVING_STR))


        fig = plt.figure()
        line_x = X_test[num_test_points-1] + (X_test[num_test_points-2]-X_test[num_test_points-3])/2
        plt.plot([line_x,line_x], Y_LIM, '--', color='gray')
        plt.text(line_x+0.1, Y_LIM[1]-Y_LIM[1]*0.02, 'Forecast', {'color': 'gray', 'fontsize': 10, 'va': 'top'})
        plt.annotate("", xy=[line_x, Y_LIM[1]-Y_LIM[1]*0.06], xytext=[line_x+0.65, Y_LIM[1]-Y_LIM[1]*0.06],
                    arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", color='gray'))

        plt.plot(X_test, Y_test, 'ok', fillstyle='none')
        plt.plot(X_test[:num_test_points], Y_test[:num_test_points], 'ok')

        plt.fill_between(X_test_ext, Y_prcntl_solely[2], Y_prcntl_solely[0], color='C0', alpha=0.2)
        p1 = plt.plot(X_test_ext, Y_prcntl_solely[1], '-', color='C0', label='Observations only')
        sub = len(X_test_ext)//10
        p2 = plt.plot(X_test_ext[::sub], Y_prcntl_solely[1][::sub], '+', color='C0', label='Observations only')
        p2_2 = plt.fill([],[], color='C0', alpha=0.2)

        plt.fill_between(X_test_ext, Y_prcntl_post[2], Y_prcntl_post[0], color='C2', alpha=0.2)
        p3 = plt.plot(X_test_ext, Y_prcntl_post[1], '-', color='C2', label='Observations + fleet prior')
        p4 = plt.plot(X_test_ext[::sub], Y_prcntl_post[1][::sub], '.', color='C2')
        p5 = plt.fill([],[], color='C2', alpha=0.2)

        plt.xlabel('Cumulative Energy (kWh)')
        plt.ylabel(PLOT_Y_LEGEND)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.grid()
        plt.legend([(p1[0], p2[0], p2_2[0]), (p3[0], p4[0], p5[0])], ['Observations only', 'Observations + fleet prior'], loc='lower left')
        plt.xlim(X_LIM)
        plt.ylim(Y_LIM)


        # fig = plt.figure()
        # plt.plot(X_test, Y_test, 'o', color='gray')
        # plt.plot(X_test[:num_test_points], Y_test[:num_test_points], 'ok')

        # plt.fill_between(X_test_ext, Y_prcntl_solely[2], Y_prcntl_solely[0], color='C0', alpha=0.2)
        # plt.plot(X_test_ext, Y_prcntl_solely[1], '-.', color='C0', linewidth=1., label='Observations only')

        # plt.fill_between(X_test_ext, Y_prcntl_post[2], Y_prcntl_post[0], color='C2', alpha=0.2)
        # plt.plot(X_test_ext, Y_prcntl_post[1], '-', color='C2', linewidth=1., label='Observations + fleet prior')

        # plt.xlabel('Cumulative Energy (kWh)')
        # plt.ylabel(PLOT_Y_LEGEND)
        # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        # plt.grid()
        # plt.legend()
        if args.save:
            fig.savefig('./figures/ensemble_test_{}pts_ens_vs_solely{}.png'.format(num_test_points, SAVING_STR))


        fig, ax = plt.subplots()
        ind = np.arange(n_batt)+1
        width = 0.35
        ax.bar(ind, w_mse_full, width, label='Prior')
        ax.bar(ind + width, w_mse_full_post, width, label='Posterior')

        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(ind)
        ax.set_xlabel('Battery #')
        ax.set_ylabel('weight')
        ax.legend()
        if args.save:
            fig.savefig('./figures/ensemble_weights_{}pts{}.png'.format(num_test_points, SAVING_STR))


        if args.save:
            # clean tf keras models from list before saving
            for batt_i in range(n_batt):
                del model_dic_list_post[batt_i]['model']     

            np.save('./training/ensemble_model_dic_list_post_{}pts{}.npy'.format(num_test_points, SAVING_STR), model_dic_list_post)


    plt.show()
