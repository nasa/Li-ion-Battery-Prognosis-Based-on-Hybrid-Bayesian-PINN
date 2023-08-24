from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops

from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class BatteryRNNCell(Layer):
    def __init__(self, initial_state=None, dt=1.0, qMobile=7600, **kwargs):
        super(BatteryRNNCell, self).__init__(**kwargs)

        self.initial_state = initial_state
        self.dt = dt
        self.qMobile = qMobile

        self.initBatteryParams()

        self.state_size  = tensor_shape.TensorShape(8)
        self.output_size = tensor_shape.TensorShape(1)

    def initBatteryParams(self):
        P = self

        P.xnMax = tf.constant(0.6, dtype=self.dtype)             # maximum mole fraction (neg electrode)
        P.xnMin = tf.constant(0, dtype=self.dtype)              # minimum mole fraction (neg electrode)
        P.xpMax = tf.constant(1.0, dtype=self.dtype)            # maximum mole fraction (pos electrode)
        P.xpMin = tf.constant(0.4, dtype=self.dtype)            # minimum mole fraction (pos electrode) -> note xn+xp=1
        P.qMax = P.qMobile/(P.xnMax-P.xnMin)    # note qMax = qn+qp
        P.Ro = tf.constant(0.117215, dtype=self.dtype)          # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)

        # Constants of nature
        P.R = tf.constant(8.3144621, dtype=self.dtype)          # universal gas constant, J/K/mol
        P.F = tf.constant(96487, dtype=self.dtype)              # Faraday's constant, C/mol

        # Li-ion parameters
        P.alpha = tf.constant(0.5, dtype=self.dtype)            # anodic/cathodic electrochemical transfer coefficient
        P.Sn = tf.constant(0.000437545, dtype=self.dtype)       # surface area (- electrode)
        P.Sp = tf.constant(0.00030962, dtype=self.dtype)        # surface area (+ electrode)
        P.kn = tf.constant(2120.96, dtype=self.dtype)           # lumped constant for BV (- electrode)
        P.kp = tf.constant(248898, dtype=self.dtype)            # lumped constant for BV (+ electrode)
        P.Vol = tf.constant(2e-5, dtype=self.dtype)             # total interior battery volume/2 (for computing concentrations)
        P.VolSFraction = tf.constant(0.1, dtype=self.dtype)     # fraction of total volume occupied by surface volume

        # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
        # same and the surface/bulk split is the same for both electrodes
        P.VolS = P.VolSFraction*P.Vol  # surface volume
        P.VolB = P.Vol - P.VolS        # bulk volume

        # set up charges (Li ions)
        P.qpMin = P.qMax*P.xpMin            # min charge at pos electrode
        P.qpMax = P.qMax*P.xpMax            # max charge at pos electrode
        P.qpSMin = P.qpMin*P.VolS/P.Vol     # min charge at surface, pos electrode
        P.qpBMin = P.qpMin*P.VolB/P.Vol     # min charge at bulk, pos electrode
        P.qpSMax = P.qpMax*P.VolS/P.Vol     # max charge at surface, pos electrode
        P.qpBMax = P.qpMax*P.VolB/P.Vol     # max charge at bulk, pos electrode
        P.qnMin = P.qMax*P.xnMin            # max charge at neg electrode
        P.qnMax = P.qMax*P.xnMax            # max charge at neg electrode
        P.qnSMax = P.qnMax*P.VolS/P.Vol     # max charge at surface, neg electrode
        P.qnBMax = P.qnMax*P.VolB/P.Vol     # max charge at bulk, neg electrode
        P.qnSMin = P.qnMin*P.VolS/P.Vol     # min charge at surface, neg electrode
        P.qnBMin = P.qnMin*P.VolB/P.Vol     # min charge at bulk, neg electrode
        P.qSMax = P.qMax*P.VolS/P.Vol       # max charge at surface (pos and neg)
        P.qBMax = P.qMax*P.VolB/P.Vol       # max charge at bulk (pos and neg)

        # time constants
        P.tDiffusion = tf.constant(7e6, dtype=self.dtype)  # diffusion time constant (increasing this causes decrease in diffusion rate)
        P.to = tf.constant(6.08671, dtype=self.dtype)      # for Ohmic voltage
        P.tsn = tf.constant(1001.38, dtype=self.dtype)     # for surface overpotential (neg)
        P.tsp = tf.constant(46.4311, dtype=self.dtype)     # for surface overpotential (pos)

        # Redlich-Kister parameters (positive electrode)
        P.U0p = tf.constant(4.03, dtype=self.dtype)

        P.BASE_Ap0 = tf.constant(-31593.7, dtype=self.dtype)
        P.BASE_Ap1 = tf.constant(0.106747, dtype=self.dtype)
        P.BASE_Ap2 = tf.constant(24606.4, dtype=self.dtype)
        P.BASE_Ap3 = tf.constant(-78561.9, dtype=self.dtype)
        P.BASE_Ap4 = tf.constant(13317.9, dtype=self.dtype)
        P.BASE_Ap5 = tf.constant(307387.0, dtype=self.dtype)
        P.BASE_Ap6 = tf.constant(84916.1, dtype=self.dtype)
        P.BASE_Ap7 = tf.constant(-1.07469e+06, dtype=self.dtype)
        P.BASE_Ap8 = tf.constant(2285.04, dtype=self.dtype)
        P.BASE_Ap9 = tf.constant(990894.0, dtype=self.dtype)
        P.BASE_Ap10 = tf.constant(283920.0, dtype=self.dtype)
        P.BASE_Ap11 = tf.constant(-161513.0, dtype=self.dtype)
        P.BASE_Ap12 = tf.constant(-469218.0, dtype=self.dtype)

        P.Ap0 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap1 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap2 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap3 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap4 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap5 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap6 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap7 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap8 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap9 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap10 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap11 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap12 = tf.Variable(1.0, dtype=self.dtype)

        # Redlich-Kister parameters (negative electrode)
        P.U0n = tf.constant(0.01, dtype=self.dtype)

        P.BASE_An0 = tf.constant(86.19, dtype=self.dtype)
        P.An0 = tf.Variable(1.0, dtype=self.dtype)

        P.An1 = tf.constant(0, dtype=self.dtype)
        P.An2 = tf.constant(0, dtype=self.dtype)
        P.An3 = tf.constant(0, dtype=self.dtype)
        P.An4 = tf.constant(0, dtype=self.dtype)
        P.An5 = tf.constant(0, dtype=self.dtype)
        P.An6 = tf.constant(0, dtype=self.dtype)
        P.An7 = tf.constant(0, dtype=self.dtype)
        P.An8 = tf.constant(0, dtype=self.dtype)
        P.An9 = tf.constant(0, dtype=self.dtype)
        P.An10 = tf.constant(0, dtype=self.dtype)
        P.An11 = tf.constant(0, dtype=self.dtype)
        P.An12 = tf.constant(0, dtype=self.dtype)

        # End of discharge voltage threshold
        P.VEOD = tf.constant(3.0, dtype=self.dtype)

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs, states):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        states = ops.convert_to_tensor(states, dtype=self.dtype)
        states = states[0,:]

        next_states = self.getNextState(states,inputs)

        output = self.getNextOutput(next_states,inputs)

        return output, [next_states]

    def getAparams(self):
        parameters = self
        Ap0 = parameters.Ap0 * parameters.BASE_Ap0
        Ap1 = parameters.Ap1 * parameters.BASE_Ap1
        Ap2 = parameters.Ap2 * parameters.BASE_Ap2
        Ap3 = parameters.Ap3 * parameters.BASE_Ap3
        Ap4 = parameters.Ap4 * parameters.BASE_Ap4
        Ap5 = parameters.Ap5 * parameters.BASE_Ap5
        Ap6 = parameters.Ap6 * parameters.BASE_Ap6
        Ap7 = parameters.Ap7 * parameters.BASE_Ap7
        Ap8 = parameters.Ap8 * parameters.BASE_Ap8
        Ap9 = parameters.Ap9 * parameters.BASE_Ap9
        Ap10 = parameters.Ap10 * parameters.BASE_Ap10
        Ap11 = parameters.Ap11 * parameters.BASE_Ap11
        Ap12 = parameters.Ap12 * parameters.BASE_Ap12

        An0 = parameters.An0 * parameters.BASE_An0

        return tf.stack([Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap9,Ap10,Ap11,Ap12,An0])


    def Vi(self, A, x, i):

        # epsilon = tf.constant(1e-16, dtype=self.dtype)
        # n_epsilon = tf.math.negative(epsilon)
        temp_x = tf.math.multiply(tf.constant(2.0, dtype=self.dtype),x)
        temp_x = tf.math.subtract(temp_x,tf.constant(1.0, dtype=self.dtype))
        pow_1 = tf.math.pow(temp_x, tf.math.add(i, tf.constant(1.0, dtype=self.dtype)))
        # pow_1 = tf.clip_by_value(tf.math.pow(temp_x, tf.math.add(i, tf.constant(1.0, dtype=self.dtype))), n_epsilon, epsilon)
        pow_2 = tf.math.pow(temp_x, tf.math.subtract(tf.constant(1.0, dtype=self.dtype), i))
        # pow_2 = tf.clip_by_value(tf.math.pow(temp_x, tf.math.subtract(tf.constant(1.0, dtype=self.dtype), i)), n_epsilon, epsilon)
        temp_2xk = tf.math.multiply(tf.math.multiply_no_nan(x,i), tf.constant(2.0, dtype=self.dtype))
        temp_2xk = tf.math.multiply_no_nan(tf.math.subtract(tf.constant(1.0, dtype=self.dtype), x), temp_2xk)
        div = tf.math.divide_no_nan(temp_2xk, pow_2)
        denum = tf.math.multiply_no_nan(tf.math.subtract(pow_1, div), A)
        ret = tf.math.divide_no_nan(denum, tf.constant(self.F, dtype=self.dtype))

        return ret
        # return A*((2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i))/self.F


    def safe_Vi(self,A,x,i):
        x_ok = tf.not_equal(x, 0.5)
        # Vi = lambda A,x,i: A*((2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i))/self.F
        safe_f = tf.zeros_like
        safe_x = tf.where(x_ok, x, tf.ones_like(x))

        return tf.where(x_ok, self.Vi(A,safe_x,i), safe_f(x))

    @tf.function
    def getNextOutput(self,X,U):
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
        P = U[:,0]

        parameters = self

        Ap0 = parameters.Ap0 * parameters.BASE_Ap0
        Ap1 = parameters.Ap1 * parameters.BASE_Ap1
        Ap2 = parameters.Ap2 * parameters.BASE_Ap2
        Ap3 = parameters.Ap3 * parameters.BASE_Ap3
        Ap4 = parameters.Ap4 * parameters.BASE_Ap4
        Ap5 = parameters.Ap5 * parameters.BASE_Ap5
        Ap6 = parameters.Ap6 * parameters.BASE_Ap6
        Ap7 = parameters.Ap7 * parameters.BASE_Ap7
        Ap8 = parameters.Ap8 * parameters.BASE_Ap8
        Ap9 = parameters.Ap9 * parameters.BASE_Ap9
        Ap10 = parameters.Ap10 * parameters.BASE_Ap10
        Ap11 = parameters.Ap11 * parameters.BASE_Ap11
        Ap12 = parameters.Ap12 * parameters.BASE_Ap12

        An0 = parameters.An0 * parameters.BASE_An0

        An1 = parameters.An1
        An2 = parameters.An2
        An3 = parameters.An3
        An4 = parameters.An4
        An5 = parameters.An5
        An6 = parameters.An6
        An7 = parameters.An7
        An8 = parameters.An8
        An9 = parameters.An9
        An10 = parameters.An10
        An11 = parameters.An11
        An12 = parameters.An12

        # Redlich-Kister expansion item
        # Vi = lambda A,x,i: A*((2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i))/parameters.F
        # Vi = self.safe_Vi
        Vi = self.Vi

        # Constraints
        Tbm = Tb-273.15
        xpS = qpS/parameters.qSMax

        Vep0 = Vi(Ap0,xpS,tf.constant(0.0, dtype=self.dtype))
        Vep1 = Vi(Ap1,xpS,tf.constant(1.0, dtype=self.dtype))
        Vep2 = Vi(Ap2,xpS,tf.constant(2.0, dtype=self.dtype))
        Vep3 = Vi(Ap3,xpS,tf.constant(3.0, dtype=self.dtype))
        Vep4 = Vi(Ap4,xpS,tf.constant(4.0, dtype=self.dtype))
        Vep5 = Vi(Ap5,xpS,tf.constant(5.0, dtype=self.dtype))
        Vep6 = Vi(Ap6,xpS,tf.constant(6.0, dtype=self.dtype))
        Vep7 = Vi(Ap7,xpS,tf.constant(7.0, dtype=self.dtype))
        Vep8 = Vi(Ap8,xpS,tf.constant(8.0, dtype=self.dtype))
        Vep9 = Vi(Ap9,xpS,tf.constant(9.0, dtype=self.dtype))
        Vep10 = Vi(Ap10,xpS,tf.constant(10.0, dtype=self.dtype))
        Vep11 = Vi(Ap11,xpS,tf.constant(11.0, dtype=self.dtype))
        Vep12 = Vi(Ap12,xpS,tf.constant(12.0, dtype=self.dtype))

        xnS = qnS/parameters.qSMax

        Ven0 = Vi(An0,xnS,tf.constant(0.0, dtype=self.dtype))
        Ven1 = Vi(An1,xnS,tf.constant(1.0, dtype=self.dtype))
        Ven2 = Vi(An2,xnS,tf.constant(2.0, dtype=self.dtype))
        Ven3 = Vi(An3,xnS,tf.constant(3.0, dtype=self.dtype))
        Ven4 = Vi(An4,xnS,tf.constant(4.0, dtype=self.dtype))
        Ven5 = Vi(An5,xnS,tf.constant(5.0, dtype=self.dtype))
        Ven6 = Vi(An6,xnS,tf.constant(6.0, dtype=self.dtype))
        Ven7 = Vi(An7,xnS,tf.constant(7.0, dtype=self.dtype))
        Ven8 = Vi(An8,xnS,tf.constant(8.0, dtype=self.dtype))
        Ven9 = Vi(An9,xnS,tf.constant(9.0, dtype=self.dtype))
        Ven10 = Vi(An10,xnS,tf.constant(10.0, dtype=self.dtype))
        Ven11 = Vi(An11,xnS,tf.constant(11.0, dtype=self.dtype))
        Ven12 = Vi(An12,xnS,tf.constant(12.0, dtype=self.dtype))

        Vep = parameters.U0p + parameters.R*Tb/parameters.F*tf.math.log((1-xpS)/xpS) + Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
        Ven = parameters.U0n + parameters.R*Tb/parameters.F*tf.math.log((1-xnS)/xnS) + Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12
        V = Vep - Ven - Vo - Vsn - Vsp

        return V

    @tf.function
    def getNextState(self,X,U):
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

        Ap0 = parameters.Ap0 * parameters.BASE_Ap0
        Ap1 = parameters.Ap1 * parameters.BASE_Ap1
        Ap2 = parameters.Ap2 * parameters.BASE_Ap2
        Ap3 = parameters.Ap3 * parameters.BASE_Ap3
        Ap4 = parameters.Ap4 * parameters.BASE_Ap4
        Ap5 = parameters.Ap5 * parameters.BASE_Ap5
        Ap6 = parameters.Ap6 * parameters.BASE_Ap6
        Ap7 = parameters.Ap7 * parameters.BASE_Ap7
        Ap8 = parameters.Ap8 * parameters.BASE_Ap8
        Ap9 = parameters.Ap9 * parameters.BASE_Ap9
        Ap10 = parameters.Ap10 * parameters.BASE_Ap10
        Ap11 = parameters.Ap11 * parameters.BASE_Ap11
        Ap12 = parameters.Ap12 * parameters.BASE_Ap12

        An0 = parameters.An0 * parameters.BASE_An0

        An1 = parameters.An1
        An2 = parameters.An2
        An3 = parameters.An3
        An4 = parameters.An4
        An5 = parameters.An5
        An6 = parameters.An6
        An7 = parameters.An7
        An8 = parameters.An8
        An9 = parameters.An9
        An10 = parameters.An10
        An11 = parameters.An11
        An12 = parameters.An12


        # Redlich-Kister expansion item
        # Vi = lambda A,x,i: A*((2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i))/parameters.F
        # Vi = self.safe_Vi
        # Vi = self.Vi

        xpS = qpS/parameters.qSMax

        # Vep0 = Vi(Ap0,xpS,tf.constant(0.0, dtype=self.dtype))
        # Vep1 = Vi(Ap1,xpS,tf.constant(1.0, dtype=self.dtype))
        # Vep2 = Vi(Ap2,xpS,tf.constant(2.0, dtype=self.dtype))
        # Vep3 = Vi(Ap3,xpS,tf.constant(3.0, dtype=self.dtype))
        # Vep4 = Vi(Ap4,xpS,tf.constant(4.0, dtype=self.dtype))
        # Vep5 = Vi(Ap5,xpS,tf.constant(5.0, dtype=self.dtype))
        # Vep6 = Vi(Ap6,xpS,tf.constant(6.0, dtype=self.dtype))
        # Vep7 = Vi(Ap7,xpS,tf.constant(7.0, dtype=self.dtype))
        # Vep8 = Vi(Ap8,xpS,tf.constant(8.0, dtype=self.dtype))
        # Vep9 = Vi(Ap9,xpS,tf.constant(9.0, dtype=self.dtype))
        # Vep10 = Vi(Ap10,xpS,tf.constant(10.0, dtype=self.dtype))
        # Vep11 = Vi(Ap11,xpS,tf.constant(11.0, dtype=self.dtype))
        # Vep12 = Vi(Ap12,xpS,tf.constant(12.0, dtype=self.dtype))

        xnS = qnS/parameters.qSMax

        # Ven0 = Vi(An0,xnS,tf.constant(0.0, dtype=self.dtype))
        # Ven1 = Vi(An1,xnS,tf.constant(1.0, dtype=self.dtype))
        # Ven2 = Vi(An2,xnS,tf.constant(2.0, dtype=self.dtype))
        # Ven3 = Vi(An3,xnS,tf.constant(3.0, dtype=self.dtype))
        # Ven4 = Vi(An4,xnS,tf.constant(4.0, dtype=self.dtype))
        # Ven5 = Vi(An5,xnS,tf.constant(5.0, dtype=self.dtype))
        # Ven6 = Vi(An6,xnS,tf.constant(6.0, dtype=self.dtype))
        # Ven7 = Vi(An7,xnS,tf.constant(7.0, dtype=self.dtype))
        # Ven8 = Vi(An8,xnS,tf.constant(8.0, dtype=self.dtype))
        # Ven9 = Vi(An9,xnS,tf.constant(9.0, dtype=self.dtype))
        # Ven10 = Vi(An10,xnS,tf.constant(10.0, dtype=self.dtype))
        # Ven11 = Vi(An11,xnS,tf.constant(11.0, dtype=self.dtype))
        # Ven12 = Vi(An12,xnS,tf.constant(12.0, dtype=self.dtype))

        # Vep = parameters.U0p + parameters.R*Tb/parameters.F*tf.math.log((1-xpS)/xpS) + Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
        # Ven = parameters.U0n + parameters.R*Tb/parameters.F*tf.math.log((1-xnS)/xnS) + Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12
        # V = Vep - Ven - Vo - Vsn - Vsp


        # Constraints
        Tbdot = tf.zeros(X.shape[0], dtype=self.dtype)
        CnBulk = qnB/parameters.VolB
        CnSurface = qnS/parameters.VolS
        CpSurface = qpS/parameters.VolS
        CpBulk = qpB/parameters.VolB
        qdotDiffusionBSn = (CnBulk-CnSurface)/parameters.tDiffusion
        qnBdot = - qdotDiffusionBSn
        Jn0 = parameters.kn*(1-xnS)**parameters.alpha*(xnS)**parameters.alpha
        qdotDiffusionBSp = (CpBulk-CpSurface)/parameters.tDiffusion
        Jp0 = parameters.kp*(1-xpS)**parameters.alpha*(xpS)**parameters.alpha
        qpBdot = - qdotDiffusionBSp
        # i = P/V
        qpSdot = i + qdotDiffusionBSp
        Jn = i/parameters.Sn
        VoNominal = i*parameters.Ro
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
        ], axis = 1)

        return XNew

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        P = self

        if self.initial_state is None:
            initial_state = tf.ones([batch_size] + tensor_shape.as_shape(self.state_size).as_list(), dtype=self.dtype) \
                 * tf.constant([[292.1, 0.0, 0.0, 0.0, P.qnBMax.numpy(), P.qnSMax.numpy(), P.qpBMin.numpy(), P.qpSMin.numpy()]], dtype=self.dtype)  # 292.1 K, about 18.95 C
        else:
            initial_state = ops.convert_to_tensor(self.initial_state, dtype=self.dtype)

        return initial_state

if __name__ == "__main__":
    # Test RNN baterry cell

    DTYPE = 'float64'
    inputs = np.ones((1,3100,1),dtype=DTYPE) * 8.0  # constant load

    cell = BatteryRNNCell(dtype=DTYPE)
    rnn = tf.keras.layers.RNN(cell, return_sequences=False, stateful=False, batch_input_shape=inputs.shape, return_state=False, dtype=DTYPE)

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
    with tf.GradientTape() as t:
        out = rnn(inputs)

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

    print(out)
    print(t.gradient(out, cell.Ap0))

    # output = rnn(inputs)

    # with writer.as_default():
    #     tf.summary.trace_export(
    #         name="my_func_trace",
    #         step=0,
    #         profiler_outdir=logdir)
