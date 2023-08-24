"""
CIF Project 2022

Physics-Informed Neural Networks for Next-Generation Aircraft

Battery Models

Matteo Corbetta
matteo.corbetta@nasa.gov
"""


from abc import abstractmethod
from imports_all import keras, tf, tensor_shape, tfops, np
import BatteryParameters as b_params




class RedlichKisterExpansion():
    
    def __init__(self, U0p=None, U0n=None, Aps=None, Ans=None, F=None, **kwargs):
        self.parameters = self.initialize(self, U0p, U0n, Aps, Ans)
        self.dtype = 'float64'
        self.F = F
        super(RedlichKisterExpansion, self).__init__(**kwargs)

    def initialize(self, U0p, U0n, Aps, Ans):
        
        params = b_params.rkexp_default()
        if U0p is not None: params['positive']['U0'] = U0p
        if Aps is not None: params['positive']['As'] = Aps
        if U0n is not None: params['negative']['U0'] = U0n
        if Ans is not None: params['negative']['As'] = Ans

        self.parameters['positive']['U0'] = params['positive']['U0']
        self.parameters['positive']['As'] = params['positive']['As']
        self.parameters['negative']['U0'] = params['negative']['U0']
        self.parameters['negative']['As'] = params['negative']['As']
        self.N = {'positive': len(self.parameters['positive']['As']), 'negative': len(self.parameters['negative']['As'])}
        return self.parameters

    def __call__(self, x, side):

        x2          = tf.math.multiply( tf.constant(2.0, dtype=self.dtype), x)       # 2x
        x2m1        = tf.math.subtract( x2, tf.constant(1.0, dtype=self.dtype))    # 2x - 1
        A_times_xfn = np.zeros((self.N[side],))

        for k in range(self.N[side]):
            x2m1_pow_kp1 = tf.math.pow(x2m1, tf.math.add(k, tf.constant(1.0, dtype=self.dtype)))    # (2x - 1)^(k+1)
            x2m1_pow_1mk = tf.math.pow(x2m1, tf.math.subtract(tf.constant(1.0, dtype=self.dtype), k))    # (2x - 1)^(1-k)
            x2k          = tf.math.multiply_no_nan(x2, k)   # 2xk   
            x2k_1mx      = tf.math.multiply_no_nan(x2k, tf.math.subtract(1, x))  # 2xk(1-x)
            x2ratio      = x2k_1mx / x2m1_pow_1mk   # 2xk(1-x) / (2x - 1)^(1-k)
            x_term       = x2m1_pow_kp1 - x2ratio   # (2x - 1)^(k+1) - 2xk(1-x) / (2x - 1)^(1-k)
            A_times_xfn[k] = tf.math.multiply_no_nan(self.parameters[side]['As'][k], x_term)    # A * ((2x - 1)^(k+1) - 2xk(1-x) / (2x - 1)^(1-k))
        A_times_xfn = tfops.convert_to_tensor(np.sum(A_times_xfn), dtype=self.dtype)

        return tf.math.divide_no_nan(A_times_xfn, tf.constant(self.F, dtype=self.dtype))


# pure-physics battery model
# ============================

class BatteryCellPhy(keras.layers.Layer):
    def __init__(self, dt=1.0, eod_threshold=3.0, init_params=None, **kwargs):
        super(BatteryCellPhy, self).__init__(**kwargs)

        
        self.dt = dt
        self.eod_th = eod_threshold

        # List of input, state, output
        self.inputs  = ['i']
        self.states  = ['tb', 'Vo', 'Vsn', 'Vsp', 'qnB', 'qnS', 'qpB', 'qpS']
        self.outputs = ['v']
        
        # Hidden state vector
        self.parameters = {key: np.nan for key in ['xnMax', 'xnMin', 'xpMax', 'xpMin', 'Ro', 'qMax', 'R', 'F', 'alpha',
                                                   'Sn', 'Sp', 'kn', 'kp', 'Volume', 'VolumeSurf', 'qMobile', 'tDiffusion',
                                                   'to', 'tsn', 'tsp']}
        self.initialize(init_params=init_params)

        self.state_size  = tensor_shape.TensorShape(8)
        self.output_size = tensor_shape.TensorShape(1)

        # Redlich-Kirster Expansion
        self.VintFn = RedlichKisterExpansion(U0p=None, U0n=None, Aps=None, Ans=None, F=self.parameters['F'])



    def initialize(self,init_params=None):
        
        # Initialize model parameters: custom (with init_params) or default
        # ===============================================================
        if init_params is not None:
            assert type(init_params)==dict, "Input 'init_params' to initial model parameters must be a dictionary."
            # This for loos is not the most efficient way to do it, but it ensures that all required parameters are passed
            for key, _ in init_params.items():  self.parameters[key] = init_params[key] 
            self.parameters['qmax'] = self.parameters['xnMax'] - self.parameters['xnMin']
        else:
            self.parameters = b_params.default()
        
        # Check parameters are initialized correctly
        # =======================================
        print('Checking parameter initialization ...', end=' ')
        for key, val in self.parameters.items():    
            assert val != np.nan, "Parameter " + key + " has not been set. Check initial parameter dictionary"
        print(' complete.')

        # Add derived parameters
        # =====================
        print('Add derived parameters ...', end=' ')
        self.parameters['VolS']   = self.parameters['VolSFraction'] * self.parameters['Vol']  # surface volume
        self.parameters['VolB']   = self.parameters['Vol']   - self.parameters['VolS']  # bulk volume
        self.parameters['qpMin']  = self.parameters['qMax']  * self.parameters['xpMin'] # min charge at pos electrode
        self.parameters['qpMax']  = self.parameters['qMax']  * self.parameters['xpMax'] # max charge at pos electrode
        self.parameters['qpSMin'] = self.parameters['qpMin'] * self.parameters['VolS'] / self.parameters['Vol'] # min charge at surface, pos electrode
        self.parameters['qpBMin'] = self.parameters['qpMin'] * self.parameters['VolB'] / self.parameters['Vol'] # min charge at bulk, pos electrode
        self.parameters['qpSMax'] = self.parameters['qpMax'] * self.parameters['VolS'] / self.parameters['Vol'] # max charge at surface, pos electrode
        self.parameters['qpBMax'] = self.parameters['qpMax'] * self.parameters['VolB'] / self.parameters['Vol'] # max charge at bulk, pos electrode
        self.parameters['qnMin']  = self.parameters['qMax']  * self.parameters['xnMin'] # max charge at neg electrode
        self.parameters['qnMax']  = self.parameters['qMax']  * self.parameters['xnMax'] # max charge at neg electrode
        self.parameters['qnSMax'] = self.parameters['qnMax'] * self.parameters['VolS'] / self.parameters['Vol'] # max charge at surface, neg electrode
        self.parameters['qnBMax'] = self.parameters['qnMax'] * self.parameters['VolB'] / self.parameters['Vol'] # max charge at bulk, neg electrode
        self.parameters['qnSMin'] = self.parameters['qnMin'] * self.parameters['VolS'] / self.parameters['Vol'] # min charge at surface, neg electrode
        self.parameters['qnBMin'] = self.parameters['qnMin'] * self.parameters['VolB'] / self.parameters['Vol'] # min charge at bulk, neg electrode
        self.parameters['qSMax']  = self.parameters['qMax']  * self.parameters['VolS'] / self.parameters['Vol'] # max charge at surface (pos and neg)
        self.parameters['qBMax']  = self.parameters['qMax']  * self.parameters['VolB'] / self.parameters['Vol'] # max charge at bulk (pos and neg)
        print(' complete.')

        return self.parameters

    def Vint_safe(self, x, side):
        x_ok   = tf.not_equal(x, 0.5)
        safe_f = tf.zeros_like()
        safe_x = tf.where(x_ok, x, tf.ones_like(x))
        return tf.where(x_ok, self.VintFn(safe_x, side), safe_f(x))

    def build(self, input_shape, **kwargs):
        self.built = True

    
    def call(self, inputs, states):
        inputs = tfops.convert_to_tensor(inputs, dtype=self.dtype)
        states = tfops.convert_to_tensor(states, dtype=self.dtype)
        states = states[0,:]

        next_states = self.getNextState(states,inputs)

        output = self.getNextOutput(next_states,inputs)

        return output, [next_states]

    @tf.function
    def getNextOutput(self, X, U):
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


# PINN battery model
# ============================
class BatteryCell(keras.layers.Layer):
    def __init__(self, 
                 q_max_model=None, 
                 R_0_model=None, 
                 curr_cum_pwh=0.0, 
                 initial_state=None, 
                 dt=1.0, 
                 qMobile=7600, 
                 mlp_trainable=True, 
                 batch_size=1, 
                 q_max_base=None, 
                 R_0_base=None, 
                 D_trainable=False, 
                 **kwargs):
        super(BatteryCell, self).__init__(**kwargs)

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
        inputs = tfops.convert_to_tensor(inputs, dtype=self.dtype)
        states = tfops.convert_to_tensor(states, dtype=self.dtype)
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
            initial_state = tfops.convert_to_tensor(self.initial_state, dtype=self.dtype)

        # tf.print('Initial state:', initial_state[:,4:])
        return initial_state