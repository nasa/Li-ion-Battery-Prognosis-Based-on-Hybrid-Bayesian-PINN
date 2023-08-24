from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from datetime import datetime

import tensorflow as tf

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_probability as tfp
tfd = tfp.distributions

class BatteryRNNCell(Layer):
    def __init__(self, q_max_model=None, R_0_model=None, curr_cum_pwh=0.0, initial_state=None, dt=1.0, qMobile=7600, mlp_trainable=True, batch_size=1, q_max_base=None, R_0_base=None, D_trainable=False, **kwargs):
        super(BatteryRNNCell, self).__init__(**kwargs)

        self.initial_state = initial_state
        self.dt = dt
        self.qMobile = qMobile
        self.q_max_base_value = q_max_base
        self.R_0_base_value = R_0_base

        self.q_max_model = q_max_model
        self.R_0_model = R_0_model
        self.curr_cum_pwh = curr_cum_pwh

        self.initBatteryParams(batch_size, D_trainable)

        self.state_size  = tensor_shape.TensorShape(8)
        self.output_size = tensor_shape.TensorShape(1)

        self.MLPp = Sequential([
            # Dense(8, activation='tanh', input_shape=(1,), dtype=self.dtype, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dense(8, activation='tanh', input_shape=(1,), dtype=self.dtype),
            Dense(4, activation='tanh', dtype=self.dtype),
            Dense(1, dtype=self.dtype),
        ], name="MLPp")

        X = np.linspace(0.0,1.0,100)

        self.MLPp.set_weights(np.load('/Users/mcorbet1/OneDrive - NASA/Code/Projects/PowertrainPINN/scripts/TF/training/mlp_initial_weights.npy',allow_pickle=True))
        # self.MLPp.set_weights(np.load('./training/mlp_initial_weight_with-I.npy',allow_pickle=True))

        Y = np.linspace(-8e-4,8e-4,100)
        self.MLPn = Sequential([Dense(1, input_shape=(1,), dtype=self.dtype)], name="MLPn")
        self.MLPn.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss="mse")
        self.MLPn.fit(X,Y, epochs=200, verbose=0)

        for layer in self.MLPp.layers:
            layer.trainable=mlp_trainable

        for layer in self.MLPn.layers:
            # layer.trainable=mlp_trainable
            layer.trainable=False

    def initBatteryParams(self, batch_size, D_trainable):
        P = self
        
        if self.q_max_base_value is None:
            self.q_max_base_value = 1.0e4

        if self.R_0_base_value is None:
            self.R_0_base_value = 1.0e1

        max_q_max = 2.3e4 / self.q_max_base_value
        initial_q_max = 1.4e4 / self.q_max_base_value

        min_R_0 = 0.05 / self.R_0_base_value
        initial_R_0 = 0.15 / self.R_0_base_value

        P.xnMax = tf.constant(0.6, dtype=self.dtype)             # maximum mole fraction (neg electrode)
        P.xnMin = tf.constant(0, dtype=self.dtype)              # minimum mole fraction (neg electrode)
        P.xpMax = tf.constant(1.0, dtype=self.dtype)            # maximum mole fraction (pos electrode)
        P.xpMin = tf.constant(0.4, dtype=self.dtype)            # minimum mole fraction (pos electrode) -> note xn+xp=1

        constraint = lambda w: w * math_ops.cast(math_ops.greater(w, 0.), self.dtype)  # contraint > 0
        # constraint_q_max = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.4, rate=0.5)
        constraint_q_max = lambda w: tf.clip_by_value(w, 0.0, max_q_max)
        constraint_R_0 = lambda w: tf.clip_by_value(w, min_R_0, 1.0)
        
        # P.qMax = P.qMobile/(P.xnMax-P.xnMin)    # note qMax = qn+qp
        # P.Ro = tf.constant(0.117215, dtype=self.dtype)          # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)
        # P.qMaxBASE = P.qMobile/(P.xnMax-P.xnMin)  # 100000
        P.qMaxBASE = tf.constant(self.q_max_base_value, dtype=self.dtype)
        P.RoBASE = tf.constant(self.R_0_base_value, dtype=self.dtype)
        # P.qMax = tf.Variable(np.ones(batch_size)*initial_q_max, constraint=constraint_q_max, dtype=self.dtype)  # init 0.1 - resp 0.1266
        # P.Ro = tf.Variable(np.ones(batch_size)*initial_R_0, constraint=constraint, dtype=self.dtype)   # init 0.15 - resp 0.117215


        if self.q_max_model is None:
            P.qMax = tf.Variable(np.ones(batch_size)*initial_q_max, constraint=constraint_q_max, dtype=self.dtype)  # init 0.1 - resp 0.1266
        else:
            P.qMax = self.q_max_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.qMaxBASE

        if self.R_0_model is None:
            P.Ro = tf.Variable(np.ones(batch_size)*initial_R_0, constraint=constraint, dtype=self.dtype)   # init 0.15 - resp 0.117215
        else:    
            P.Ro = self.R_0_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.RoBASE

        # Constants of nature
        P.R = tf.constant(8.3144621, dtype=self.dtype)          # universal gas constant, J/K/mol
        P.F = tf.constant(96487, dtype=self.dtype)              # Faraday's constant, C/mol

        # Li-ion parameters
        P.alpha = tf.constant(0.5, dtype=self.dtype)            # anodic/cathodic electrochemical transfer coefficient
        # P.Sn = tf.constant(0.000437545, dtype=self.dtype)       # surface area (- electrode)
        # P.Sp = tf.constant(0.00030962, dtype=self.dtype)        # surface area (+ electrode)
        # P.kn = tf.constant(2120.96, dtype=self.dtype)           # lumped constant for BV (- electrode)
        # P.kp = tf.constant(248898, dtype=self.dtype)            # lumped constant for BV (+ electrode)
        # P.Vol = tf.constant(2e-5, dtype=self.dtype)             # total interior battery volume/2 (for computing concentrations)
        P.VolSFraction = tf.constant(0.1, dtype=self.dtype)     # fraction of total volume occupied by surface volume

        P.Sn = tf.constant(2e-4, dtype=self.dtype)       # surface area (- electrode)
        P.Sp = tf.constant(2e-4, dtype=self.dtype)        # surface area (+ electrode)
        P.kn = tf.constant(2e4, dtype=self.dtype)           # lumped constant for BV (- electrode)
        P.kp = tf.constant(2e4, dtype=self.dtype)            # lumped constant for BV (+ electrode)
        P.Vol = tf.constant(2.2e-5, dtype=self.dtype)    

        # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
        # same and the surface/bulk split is the same for both electrodes
        P.VolS = P.VolSFraction*P.Vol  # surface volume
        P.VolB = P.Vol - P.VolS        # bulk volume

        # set up charges (Li ions)
        P.qpMin = P.qMax*P.qMaxBASE*P.xpMin            # min charge at pos electrode
        P.qpMax = P.qMax*P.qMaxBASE*P.xpMax            # max charge at pos electrode
        P.qpSMin = P.qpMin*P.VolS/P.Vol     # min charge at surface, pos electrode
        P.qpBMin = P.qpMin*P.VolB/P.Vol     # min charge at bulk, pos electrode
        P.qpSMax = P.qpMax*P.VolS/P.Vol     # max charge at surface, pos electrode
        P.qpBMax = P.qpMax*P.VolB/P.Vol     # max charge at bulk, pos electrode
        P.qnMin = P.qMax*P.qMaxBASE*P.xnMin            # max charge at neg electrode
        P.qnMax = P.qMax*P.qMaxBASE*P.xnMax            # max charge at neg electrode
        P.qnSMax = P.qnMax*P.VolS/P.Vol     # max charge at surface, neg electrode
        P.qnBMax = P.qnMax*P.VolB/P.Vol     # max charge at bulk, neg electrode
        P.qnSMin = P.qnMin*P.VolS/P.Vol     # min charge at surface, neg electrode
        P.qnBMin = P.qnMin*P.VolB/P.Vol     # min charge at bulk, neg electrode
        P.qSMax = P.qMax*P.qMaxBASE*P.VolS/P.Vol       # max charge at surface (pos and neg)
        P.qBMax = P.qMax*P.qMaxBASE*P.VolB/P.Vol       # max charge at bulk (pos and neg)

        # time constants
        # P.tDiffusion = tf.constant(7e6, dtype=self.dtype)  # diffusion time constant (increasing this causes decrease in diffusion rate)
        # P.tDiffusion = tf.constant(7e6, dtype=self.dtype)  # diffusion time constant (increasing this causes decrease in diffusion rate)
        P.tDiffusion = tf.Variable(7e6, trainable=D_trainable, dtype=self.dtype)  # diffusion time constant (increasing this causes decrease in diffusion rate)
        # P.to = tf.constant(6.08671, dtype=self.dtype)      # for Ohmic voltage
        # P.tsn = tf.constant(1001.38, dtype=self.dtype)     # for surface overpotential (neg)
        # P.tsp = tf.constant(46.4311, dtype=self.dtype)     # for surface overpotential (pos)

        P.to = tf.constant(10.0, dtype=self.dtype)      # for Ohmic voltage
        P.tsn = tf.constant(90.0, dtype=self.dtype)     # for surface overpotential (neg)
        P.tsp = tf.constant(90.0, dtype=self.dtype)     # for surface overpotential (pos)

        # Redlich-Kister parameters (positive electrode)
        P.U0p = tf.constant(4.03, dtype=self.dtype)
        # P.U0p = tf.Variable(4.03, dtype=self.dtype)

        # Redlich-Kister parameters (negative electrode)
        P.U0n = tf.constant(0.01, dtype=self.dtype)

        # End of discharge voltage threshold
        P.VEOD = tf.constant(3.0, dtype=self.dtype)

    def build(self, input_shape, **kwargs):
        self.built = True

    @tf.function
    def call(self, inputs, states, training=None):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        states = ops.convert_to_tensor(states, dtype=self.dtype)
        states = states[0,:]

        next_states = self.getNextState(states,inputs,training)

        output = self.getNextOutput(next_states,inputs,training)

        return output, [next_states]

    def getAparams(self):
        return self.MLPp.get_weights()

    # @tf.function
    def getNextOutput(self,X,U,training):
        # OutputEqn   Compute the outputs of the battery model
        #
        #   Z = OutputEqn(parameters,t,X,U,N) computes the outputs of the battery
        #   model given the parameters structure, time, the states, inputs, and
        #   sensor noise. The function is vectorized, so if the function inputs are
        #   matrices, the funciton output will be a matrix, with the rows being the
        #   variables and the columns the samples.
        #
        #   Copyright (c)�2016 United States Government as represented by the
        #   Administrator of the National Aeronautics and Space Administration.
        #   No copyright is claimed in the United States under Title 17, U.S.
        #   Code. All Other Rights Reserved.

        # Extract states
        Tb = X[:,0]
        Vo = X[:,1]
        Vsn = X[:,2]
        Vsp = X[:,3]
        qnB = X[:,4]
        qnS = X[:,5]
        qpB = X[:,6]
        qpS = X[:,7]

        # Extract inputs
        # P = U[:,0]
        i = U[:,0]

        parameters = self

        qSMax = (parameters.qMax * parameters.qMaxBASE) * parameters.VolS/parameters.Vol

        # Constraints
        Tbm = Tb-273.15
        xpS = qpS/qSMax
        xnS = qnS/qSMax

        # tf.print('qpS:', qpS)
        # tf.print('xpS:', xpS)

        # VepMLP = self.MLPp(tf.expand_dims(xpS,1))[:,0] * self.MLPpFACTOR
        # VenMLP = self.MLPn(tf.expand_dims(xnS,1))[:,0] * self.MLPnFACTOR

        # VepMLP = self.MLPp(tf.stack([xpS, i],1))[:,0]
        VepMLP = self.MLPp(tf.expand_dims(xpS,1))[:,0]
        VenMLP = self.MLPn(tf.expand_dims(xnS,1))[:,0]

        # if training:
        safe_log_p = tf.clip_by_value((1-xpS)/xpS,1e-18,1e+18)
        safe_log_n = tf.clip_by_value((1-xnS)/xnS,1e-18,1e+18)
        # else:
        #     safe_log_p = (1.0-xpS)/xpS
        #     safe_log_n = (1.0-xnS)/xnS

        Vep = parameters.U0p + parameters.R*Tb/parameters.F*tf.math.log(safe_log_p) + VepMLP
        Ven = parameters.U0n + parameters.R*Tb/parameters.F*tf.math.log(safe_log_n) + VenMLP
        V = Vep - Ven - Vo - Vsn - Vsp

        return tf.expand_dims(V,1, name="output")

    # @tf.function
    def getNextState(self,X,U,training):
        # StateEqn   Compute the new states of the battery model
        #
        #   XNew = StateEqn(parameters,t,X,U,N,dt) computes the new states of the
        #   battery model given the parameters strcucture, the current time, the
        #   current states, inputs, process noise, and the sampling time.
        #
        #   Copyright (c)�2016 United States Government as represented by the
        #   Administrator of the National Aeronautics and Space Administration.
        #   No copyright is claimed in the United States under Title 17, U.S.
        #   Code. All Other Rights Reserved.

        # Extract states
        Tb = X[:,0]
        Vo = X[:,1]
        Vsn = X[:,2]
        Vsp = X[:,3]
        qnB = X[:,4]
        qnS = X[:,5]
        qpB = X[:,6]
        qpS = X[:,7]

        # Extract inputs
        # P = U[:,0]
        i = U[:,0]

        parameters = self

        qSMax = (parameters.qMax * parameters.qMaxBASE) * parameters.VolS/parameters.Vol

        # xpS = qpS/parameters.qSMax
        # xnS = qnS/parameters.qSMax

        # safe values for mole frac when training
        # if training:
        xpS = tf.clip_by_value(qpS/qSMax,1e-18,1.0)
        xnS = tf.clip_by_value(qnS/qSMax,1e-18,1.0)
        Jn0 = 1e-18 + parameters.kn*(1-xnS)**parameters.alpha*(xnS)**parameters.alpha
        Jp0 = 1e-18 + parameters.kp*(1-xpS)**parameters.alpha*(xpS)**parameters.alpha
        # else:
        #     xpS = qpS/qSMax
        #     xnS = qnS/qSMax
        #     Jn0 = parameters.kn*(1-xnS)**parameters.alpha*(xnS)**parameters.alpha
        #     Jp0 = parameters.kp*(1-xpS)**parameters.alpha*(xpS)**parameters.alpha

        # Constraints
        Tbdot = tf.zeros(X.shape[0], dtype=self.dtype)
        CnBulk = qnB/parameters.VolB
        CnSurface = qnS/parameters.VolS
        CpSurface = qpS/parameters.VolS
        CpBulk = qpB/parameters.VolB
        qdotDiffusionBSn = (CnBulk-CnSurface)/parameters.tDiffusion
        qnBdot = - qdotDiffusionBSn
        qdotDiffusionBSp = (CpBulk-CpSurface)/parameters.tDiffusion
        qpBdot = - qdotDiffusionBSp
        # i = P/V
        qpSdot = i + qdotDiffusionBSp
        Jn = i/parameters.Sn
        VoNominal = i*parameters.Ro*parameters.RoBASE
        Jp = i/parameters.Sp
        qnSdot = qdotDiffusionBSn - i
        VsnNominal = parameters.R*Tb/parameters.F/parameters.alpha*tf.math.asinh(Jn/(2*Jn0))
        Vodot = (VoNominal-Vo)/parameters.to
        VspNominal = parameters.R*Tb/parameters.F/parameters.alpha*tf.math.asinh(Jp/(2*Jp0))
        Vsndot = (VsnNominal-Vsn)/parameters.tsn
        Vspdot = (VspNominal-Vsp)/parameters.tsp

        dt = self.dt
        # Update state
        XNew = tf.stack([
            Tb + Tbdot*dt,
            Vo + Vodot*dt,
            Vsn + Vsndot*dt,
            Vsp + Vspdot*dt,
            qnB + qnBdot*dt,
            qnS + qnSdot*dt,
            qpB + qpBdot*dt,
            qpS + qpSdot*dt
        ], axis = 1, name='next_states')

        return XNew

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        P = self

        if self.q_max_model is not None:
            # P.qMax = self.q_max_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.qMaxBASE
            P.qMax = tf.concat([self.q_max_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.qMaxBASE for _ in range(100)], axis=0)

        if self.R_0_model is not None: 
            # P.Ro = self.R_0_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.RoBASE
            P.Ro = tf.concat([self.R_0_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.RoBASE for _ in range(100)], axis=0)

        qpMin = P.qMax*P.qMaxBASE*P.xpMin            # min charge at pos electrode
        qpSMin = qpMin*P.VolS/P.Vol     # min charge at surface, pos electrode
        qpBMin = qpMin*P.VolB/P.Vol     # min charge at bulk, pos electrode
        qnMax = P.qMax*P.qMaxBASE*P.xnMax            # max charge at neg electrode
        qnSMax = qnMax*P.VolS/P.Vol     # max charge at surface, neg electrode
        qnBMax = qnMax*P.VolB/P.Vol     # max charge at bulk, neg electrode


        if self.initial_state is None:
            # if P.qMax.shape[0]==1:
            #     initial_state = tf.ones([batch_size] + tensor_shape.as_shape(self.state_size).as_list(), dtype=self.dtype) \
            #         * tf.stack([tf.constant(292.1, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype), qnBMax[0], qnSMax[0], qpBMin[0], qpSMin[0]])  # 292.1 K, about 18.95 C
            # else:
            initial_state_0_3 = tf.ones([P.qMax.shape[0], 4], dtype=self.dtype) \
                * tf.constant([292.1, 0.0, 0.0, 0.0], dtype=self.dtype)
            initial_state = tf.concat([initial_state_0_3, tf.expand_dims(qnBMax, axis=1), tf.expand_dims(qnSMax, axis=1), tf.expand_dims(qpBMin, axis=1), tf.expand_dims(qpSMin, axis=1)], axis=1)
        else:
            initial_state = ops.convert_to_tensor(self.initial_state, dtype=self.dtype)

        # tf.print('Initial state:', initial_state[:,4:])
        return initial_state

if __name__ == "__main__":
    # Test RNN baterry cell

    DTYPE = 'float32'
    dt = 10.0
    # inputs = np.hstack([np.zeros((1,50,1),dtype=DTYPE), np.ones((1,100,1),dtype=DTYPE)]) * 2.0  # constant load
    # inputs = np.array([1.003, 0.999, 1.   , 1.   , 1.   , 0.999, 1.   , 1.   , 1.   ,
    #    1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 0.999,
    #    1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ,
    #    1.   , 0.999, 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ,
    #    1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ,
    #    1.   , 1.   , 1.   , 1.001, 1.001, 0.999, 1.   , 1.   , 1.   ,
    #    1.   , 0.999, 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ,
    #    1.   , 1.   , 1.   , 1.001, 1.   , 1.   , 1.   , 1.   , 1.   ,
    #    1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ,
    #    1.   , 0.999, 1.   , 1.   , 0.999, 1.   , 1.   , 1.   , 1.   ,
    #    1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 0.999, 1.   ,
    #    1.   ])[np.newaxis, :, np.newaxis]
    inputs = np.ones((1000,800,1))
    # inputs = np.zeros((30,700,1),dtype=DTYPE)  # constant load

    cell = BatteryRNNCell(dtype=DTYPE, dt=dt, batch_size=inputs.shape[0])
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, batch_input_shape=inputs.shape, return_state=False, dtype=DTYPE)

    cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))

    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = 'logs/func/%s' % stamp
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=True)

    outputs = []
    H = []
    grads = []

    # tf.debugging.enable_check_numerics()

    # test cell output and gradient calc
    # with tf.GradientTape(persistent=True) as t:
    # with tf.GradientTape() as t:
        # out = rnn(inputs)

        # for i in range(500):
        #     if i==0:
        #         out, states = cell(inputs[:,0, :], [cell.get_initial_state(batch_size=inputs.shape[0])])
        #     else:
        #         out, states = cell(inputs[:,i, :], states)

        #     with t.stop_recording():
        #         o = out.numpy()
        #         s = states[0].numpy()
        #         g = t.gradient(out, cell.Ap0).numpy()
        #         outputs.append(o)
        #         H.append(s)
        #         grads.append(g)
        #         print("t:{}, V:{}, dV_dAp0:{}".format(i, o, g))
        #         print("states:{}".format(s))

    # print(out)
    # print(t.gradient(out, cell.MLPp.variables[0]))

    # out, states = cell(inputs[:,0, :], [cell.get_initial_state(batch_size=inputs.shape[0])])

    output = rnn(inputs)[:,:,0].numpy()

    mean = np.quantile(output, 0.5, 0)
    lb = np.quantile(output, 0.025, 0)
    ub = np.quantile(output, 0.975, 0)
    std = output.std(axis=0)

    # plt.plot(output[:,:,0].numpy().T)
    plt.fill_between(np.arange(inputs.shape[1]), 
                 ub, 
                 lb, 
                 alpha=0.5)
    plt.plot(mean)

    plt.grid()
    plt.show()

    # with writer.as_default():
    #     tf.summary.trace_export(
    #         name="my_func_trace",
    #         step=0,
    #         profiler_outdir=logdir)
