from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops

from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class BatteryRNNCell(Layer):
    def __init__(self, initial_state=None, dt=1, qMobile=7600, **kwargs):
        super(BatteryRNNCell, self).__init__(**kwargs)

        self.initial_state = initial_state
        self.dt = dt
        self.qMobile = qMobile

        self.initBatteryParams()

        self.state_size  = tensor_shape.TensorShape(8)
        self.output_size = tensor_shape.TensorShape(2)

    def initBatteryParams(self):
        P = self
        P.xnMax = 0.6            # maximum mole fraction (neg electrode)
        P.xnMin = 0              # minimum mole fraction (neg electrode)
        P.xpMax = 1.0            # maximum mole fraction (pos electrode)
        P.xpMin = 0.4            # minimum mole fraction (pos electrode) -> note xn+xp=1
        P.qMax = P.qMobile/(P.xnMax-P.xnMin)    # note qMax = qn+qp
        P.Ro = 0.117215          # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)

        # Constants of nature
        P.R = 8.3144621          # universal gas constant, J/K/mol
        P.F = 96487              # Faraday's constant, C/mol

        # Li-ion parameters
        P.alpha = 0.5            # anodic/cathodic electrochemical transfer coefficient
        P.Sn = 0.000437545       # surface area (- electrode)
        P.Sp = 0.00030962        # surface area (+ electrode)
        P.kn = 2120.96           # lumped constant for BV (- electrode)
        P.kp = 248898            # lumped constant for BV (+ electrode)
        P.Vol = 2e-5             # total interior battery volume/2 (for computing concentrations)
        P.VolSFraction = 0.1     # fraction of total volume occupied by surface volume

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
        P.tDiffusion = 7e6  # diffusion time constant (increasing this causes decrease in diffusion rate)
        P.to = 6.08671      # for Ohmic voltage
        P.tsn = 1001.38     # for surface overpotential (neg)
        P.tsp = 46.4311     # for surface overpotential (pos)

        # Redlich-Kister parameters (positive electrode)
        P.U0p = 4.03
        P.Ap0 = -31593.7
        P.Ap1 = 0.106747
        P.Ap2 = 24606.4
        P.Ap3 = -78561.9
        P.Ap4 = 13317.9
        P.Ap5 = 307387
        P.Ap6 = 84916.1
        P.Ap7 = -1.07469e+06
        P.Ap8 = 2285.04
        P.Ap9 = 990894
        P.Ap10 = 283920
        P.Ap11 = -161513
        P.Ap12 = -469218

        # Redlich-Kister parameters (negative electrode)
        P.U0n = 0.01
        P.An0 = 86.19
        P.An1 = 0
        P.An2 = 0
        P.An3 = 0
        P.An4 = 0
        P.An5 = 0
        P.An6 = 0
        P.An7 = 0
        P.An8 = 0
        P.An9 = 0
        P.An10 = 0
        P.An11 = 0
        P.An12 = 0

        # End of discharge voltage threshold
        P.VEOD = 3.0

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs, states):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        states = ops.convert_to_tensor(states, dtype=self.dtype)
        states = states[0,:]

        next_states = self.getNextState(states,inputs)

        output = self.getNextOutput(next_states,inputs)

        return output, [next_states]

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
        # P = U(1,:)
        P = U[:,0]

        parameters = self
        # Constraints
        Tbm = Tb-273.15
        xpS = qpS/parameters.qSMax
        Vep3 = parameters.Ap3*((2*xpS-1)**(3+1) - (2*xpS*3*(1-xpS))/(2*xpS-1)**(1-3))/parameters.F
        Vep8 = parameters.Ap8*((2*xpS-1)**(8+1) - (2*xpS*8*(1-xpS))/(2*xpS-1)**(1-8))/parameters.F
        Vep6 = parameters.Ap6*((2*xpS-1)**(6+1) - (2*xpS*6*(1-xpS))/(2*xpS-1)**(1-6))/parameters.F
        Vep5 = parameters.Ap5*((2*xpS-1)**(5+1) - (2*xpS*5*(1-xpS))/(2*xpS-1)**(1-5))/parameters.F
        Vep10 = parameters.Ap10*((2*xpS-1)**(10+1) - (2*xpS*10*(1-xpS))/(2*xpS-1)**(1-10))/parameters.F
        Vep9 = parameters.Ap9*((2*xpS-1)**(9+1) - (2*xpS*9*(1-xpS))/(2*xpS-1)**(1-9))/parameters.F
        Vep12 = parameters.Ap12*((2*xpS-1)**(12+1) - (2*xpS*12*(1-xpS))/(2*xpS-1)**(1-12))/parameters.F
        Vep4 = parameters.Ap4*((2*xpS-1)**(4+1) - (2*xpS*4*(1-xpS))/(2*xpS-1)**(1-4))/parameters.F
        Vep11 = parameters.Ap11*((2*xpS-1)**(11+1) - (2*xpS*11*(1-xpS))/(2*xpS-1)**(1-11))/parameters.F
        Vep2 = parameters.Ap2*((2*xpS-1)**(2+1) - (2*xpS*2*(1-xpS))/(2*xpS-1)**(1-2))/parameters.F
        Vep7 = parameters.Ap7*((2*xpS-1)**(7+1) - (2*xpS*7*(1-xpS))/(2*xpS-1)**(1-7))/parameters.F
        Vep0 = parameters.Ap0*((2*xpS-1)**(0+1))/parameters.F
        Vep1 = parameters.Ap1*((2*xpS-1)**(1+1) - (2*xpS*1*(1-xpS))/(2*xpS-1)**(1-1))/parameters.F
        xnS = qnS/parameters.qSMax
        Ven5 = parameters.An5*((2*xnS-1)**(5+1) - (2*xnS*5*(1-xnS))/(2*xnS-1)**(1-5))/parameters.F
        Ven1 = parameters.An1*((2*xnS-1)**(1+1) - (2*xnS*1*(1-xnS))/(2*xnS-1)**(1-1))/parameters.F
        Ven10 = parameters.An10*((2*xnS-1)**(10+1) - (2*xnS*10*(1-xnS))/(2*xnS-1)**(1-10))/parameters.F
        Ven7 = parameters.An7*((2*xnS-1)**(7+1) - (2*xnS*7*(1-xnS))/(2*xnS-1)**(1-7))/parameters.F
        Ven2 = parameters.An2*((2*xnS-1)**(2+1) - (2*xnS*2*(1-xnS))/(2*xnS-1)**(1-2))/parameters.F
        Ven8 = parameters.An8*((2*xnS-1)**(8+1) - (2*xnS*8*(1-xnS))/(2*xnS-1)**(1-8))/parameters.F
        Ven4 = parameters.An4*((2*xnS-1)**(4+1) - (2*xnS*4*(1-xnS))/(2*xnS-1)**(1-4))/parameters.F
        Ven3 = parameters.An3*((2*xnS-1)**(3+1) - (2*xnS*3*(1-xnS))/(2*xnS-1)**(1-3))/parameters.F
        Vep = parameters.U0p + parameters.R*Tb/parameters.F*tf.math.log((1-xpS)/xpS) + Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
        Ven0 = parameters.An0*((2*xnS-1)**(0+1))/parameters.F
        Ven11 = parameters.An11*((2*xnS-1)**(11+1) - (2*xnS*11*(1-xnS))/(2*xnS-1)**(1-11))/parameters.F
        Ven12 = parameters.An12*((2*xnS-1)**(12+1) - (2*xnS*12*(1-xnS))/(2*xnS-1)**(1-12))/parameters.F
        Ven6 = parameters.An6*((2*xnS-1)**(6+1) - (2*xnS*6*(1-xnS))/(2*xnS-1)**(1-6))/parameters.F
        Ven9 = parameters.An9*((2*xnS-1)**(9+1) - (2*xnS*9*(1-xnS))/(2*xnS-1)**(1-9))/parameters.F
        Ven = parameters.U0n + parameters.R*Tb/parameters.F*tf.math.log((1-xnS)/xnS) + Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12
        V = Vep - Ven - Vo - Vsn - Vsp
        # Vm = V

        # set outputs
        Z = tf.stack([ Tbm, V ], axis = 1)

        # Add sensor noise
        # Z = Z + N
        return Z

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
        # P = U(1,:)
        P = U[:,0]

        parameters = self
        # Constraints
        Tbdot = 0
        CnBulk = qnB/parameters.VolB
        CnSurface = qnS/parameters.VolS
        CpSurface = qpS/parameters.VolS
        xnS = qnS/parameters.qSMax
        Ven5 = parameters.An5*((2*xnS-1)**(5+1) - (2*xnS*5*(1-xnS))/(2*xnS-1)**(1-5))/parameters.F
        xpS = qpS/parameters.qSMax
        Vep3 = parameters.Ap3*((2*xpS-1)**(3+1) - (2*xpS*3*(1-xpS))/(2*xpS-1)**(1-3))/parameters.F
        Vep12 = parameters.Ap12*((2*xpS-1)**(12+1) - (2*xpS*12*(1-xpS))/(2*xpS-1)**(1-12))/parameters.F
        Vep4 = parameters.Ap4*((2*xpS-1)**(4+1) - (2*xpS*4*(1-xpS))/(2*xpS-1)**(1-4))/parameters.F
        Vep11 = parameters.Ap11*((2*xpS-1)**(11+1) - (2*xpS*11*(1-xpS))/(2*xpS-1)**(1-11))/parameters.F
        Vep2 = parameters.Ap2*((2*xpS-1)**(2+1) - (2*xpS*2*(1-xpS))/(2*xpS-1)**(1-2))/parameters.F
        Vep7 = parameters.Ap7*((2*xpS-1)**(7+1) - (2*xpS*7*(1-xpS))/(2*xpS-1)**(1-7))/parameters.F
        CpBulk = qpB/parameters.VolB
        Vep8 = parameters.Ap8*((2*xpS-1)**(8+1) - (2*xpS*8*(1-xpS))/(2*xpS-1)**(1-8))/parameters.F
        qdotDiffusionBSn = (CnBulk-CnSurface)/parameters.tDiffusion
        qnBdot = - qdotDiffusionBSn
        Jn0 = parameters.kn*(1-xnS)**parameters.alpha*(xnS)**parameters.alpha
        Ven3 = parameters.An3*((2*xnS-1)**(3+1) - (2*xnS*3*(1-xnS))/(2*xnS-1)**(1-3))/parameters.F
        qdotDiffusionBSp = (CpBulk-CpSurface)/parameters.tDiffusion
        Ven0 = parameters.An0*((2*xnS-1)**(0+1))/parameters.F
        Jp0 = parameters.kp*(1-xpS)**parameters.alpha*(xpS)**parameters.alpha
        Ven10 = parameters.An10*((2*xnS-1)**(10+1) - (2*xnS*10*(1-xnS))/(2*xnS-1)**(1-10))/parameters.F
        Ven7 = parameters.An7*((2*xnS-1)**(7+1) - (2*xnS*7*(1-xnS))/(2*xnS-1)**(1-7))/parameters.F
        Ven2 = parameters.An2*((2*xnS-1)**(2+1) - (2*xnS*2*(1-xnS))/(2*xnS-1)**(1-2))/parameters.F
        Ven11 = parameters.An11*((2*xnS-1)**(11+1) - (2*xnS*11*(1-xnS))/(2*xnS-1)**(1-11))/parameters.F
        Ven8 = parameters.An8*((2*xnS-1)**(8+1) - (2*xnS*8*(1-xnS))/(2*xnS-1)**(1-8))/parameters.F
        Ven12 = parameters.An12*((2*xnS-1)**(12+1) - (2*xnS*12*(1-xnS))/(2*xnS-1)**(1-12))/parameters.F
        Ven1 = parameters.An1*((2*xnS-1)**(1+1) - (2*xnS*1*(1-xnS))/(2*xnS-1)**(1-1))/parameters.F
        Ven4 = parameters.An4*((2*xnS-1)**(4+1) - (2*xnS*4*(1-xnS))/(2*xnS-1)**(1-4))/parameters.F
        Ven6 = parameters.An6*((2*xnS-1)**(6+1) - (2*xnS*6*(1-xnS))/(2*xnS-1)**(1-6))/parameters.F
        Ven9 = parameters.An9*((2*xnS-1)**(9+1) - (2*xnS*9*(1-xnS))/(2*xnS-1)**(1-9))/parameters.F
        Vep0 = parameters.Ap0*((2*xpS-1)**(0+1))/parameters.F
        Vep5 = parameters.Ap5*((2*xpS-1)**(5+1) - (2*xpS*5*(1-xpS))/(2*xpS-1)**(1-5))/parameters.F
        Vep6 = parameters.Ap6*((2*xpS-1)**(6+1) - (2*xpS*6*(1-xpS))/(2*xpS-1)**(1-6))/parameters.F
        Vep1 = parameters.Ap1*((2*xpS-1)**(1+1) - (2*xpS*1*(1-xpS))/(2*xpS-1)**(1-1))/parameters.F
        Vep10 = parameters.Ap10*((2*xpS-1)**(10+1) - (2*xpS*10*(1-xpS))/(2*xpS-1)**(1-10))/parameters.F
        Vep9 = parameters.Ap9*((2*xpS-1)**(9+1) - (2*xpS*9*(1-xpS))/(2*xpS-1)**(1-9))/parameters.F
        qpBdot = - qdotDiffusionBSp
        Ven = parameters.U0n + parameters.R*Tb/parameters.F*tf.math.log((1-xnS)/xnS) + Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12
        Vep = parameters.U0p + parameters.R*Tb/parameters.F*tf.math.log((1-xpS)/xpS) + Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
        V = Vep - Ven - Vo - Vsn - Vsp
        i = P/V
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

        # Add process noise
        # XNew = XNew + dt*N
        return XNew

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        P = self

        if self.initial_state is None:
            initial_state = tf.constant([[292.1, 0.0, 0.0, 0.0, P.qnBMax, P.qnSMax, P.qpBMin, P.qpSMin]])  # 292.1 K, about 18.95 C
            # P.x0.Tb = 292.1 
            # P.x0.Vo = 0
            # P.x0.Vsn = 0
            # P.x0.Vsp = 0
            # P.x0.qnB = P.qnBMax
            # P.x0.qnS = P.qnSMax
            # P.x0.qpB = P.qpBMin
            # P.x0.qpS = P.qpSMin
        else:
            initial_state = ops.convert_to_tensor(self.initial_state, dtype=self.dtype)

        return initial_state

if __name__ == "__main__":
    # Test RNN baterry cell

    rnn = tf.keras.layers.RNN(BatteryRNNCell(), return_sequences=True)

    inputs = np.ones((1,3100,1), dtype='float32') * 8.0  # constant load

    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = 'logs/func/%s' % stamp
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=True)

    output = rnn(inputs)

    # with writer.as_default():
    #     tf.summary.trace_export(
    #         name="my_func_trace",
    #         step=0,
    #         profiler_outdir=logdir)

    print(output.shape)
    plt.plot(output.numpy()[0,:,1])
    plt.show()