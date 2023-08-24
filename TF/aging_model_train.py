# %% imports
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

matplotlib.rc('font', size=14)

# %% MODELs def and func
# -------------------------------------
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    # c = np.log(np.expm1(1.))
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
            tfp.layers.DistributionLambda(lambda t: 
                tfd.Normal(loc=self.loc , scale=self.scale),
            ),
        ])


class MeanStdVar(tf.keras.layers.Layer):
    def __init__(self, prior_loc=0, prior_scale=2, batch_size=1):
        super(MeanStdVar, self).__init__()
        self.mean = tf.keras.Sequential([
            tfp.layers.DenseVariational(4, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(2, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size, activation='elu'),
            tfp.layers.DenseVariational(1, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size)
        ])
        self.std = tf.keras.Sequential([
            tfp.layers.DenseVariational(1, posterior_mean_field, PriorDist(prior_loc, prior_scale).prior_fn, kl_weight=1/batch_size)
        ])
    
    def build(self, input_shape):
        super(MeanStdVar, self).__init__()

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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=False, action="store_true" , help="Save results")
    parser.add_argument("--r0", default=False, action="store_true" , help="Train for R_0 [q_max otherwhise]")
    args = parser.parse_args()

    # load data
    from aging_model_data import *

    PLOT_Y_LEGEND = r'$q_{MAX}$'
    SAVING_STR = ''
    Y_ref = q_max_ref
    Y_LIM = Y_LIM_q_max
    Y_MAX = Y_MAX_q_max
    Y_test = Y_test_q_max

    SAVE_DATA_PATH = SAVE_DATA_PATH_q_max

    # model_dic_list = model_dic_list_q_max

    if args.r0:
        PLOT_Y_LEGEND = r'$R_0$'
        SAVING_STR = '_R_0'
        Y_ref = R_0_ref
        Y_LIM = Y_LIM_R_0
        Y_MAX = Y_MAX_R_0
        Y_test = Y_test_R_0

        SAVE_DATA_PATH = SAVE_DATA_PATH_R_0

        # model_dic_list = model_dic_list_R_0

    # TRAIN models for q_max
    model_dic_list = []
    for batt_i in range(n_batt):
        model_dic = {
            'batt_i': batt_i
        }
        print('')
        print('* * * Building batt model - {}/{} * * *'.format(batt_i+1, n_batt))

        X = cum_kWh_ref[batt_i]
        X_norm = X / X_MAX
        Y = Y_ref[batt_i]
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
        

    # VIZ results
    for batt_i in range(n_batt):
        fig = plt.figure()
        X = cum_kWh_ref[batt_i]
        X_norm = X / X_MAX
        Y = Y_ref[batt_i]
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
        plt.xlabel('Cumulative Energy (kWh)')
        plt.ylabel(PLOT_Y_LEGEND)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        # plt.xlim([0,3.5])
        # plt.ylim([6000,14000])
        plt.grid()

    plt.show()

    if not args.save:
        SAVE = input("Save results (Y/N): ").lower()=='y'
    else:
        SAVE = True
    
    if SAVE:
        print("* * SAVING RESULTS IN {} * *".format(SAVE_DATA_PATH))
        # clean tf keras models from list before saving
        for batt_i in range(n_batt):
            del model_dic_list[batt_i]['model']

        np.save(SAVE_DATA_PATH, model_dic_list)
