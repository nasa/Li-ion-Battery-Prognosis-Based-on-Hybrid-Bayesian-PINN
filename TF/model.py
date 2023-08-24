import numpy as np
from time import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

DTYPE = 'float64'

from BatteryRNNCell import BatteryRNNCell
from BatteryRNNCell_mlp import BatteryRNNCell as BatteryRNNCellMLP

def get_model(batch_input_shape=None, return_sequences=True, stateful=False, dtype=DTYPE, dt=1.0, mlp=False, mlp_trainable=True, share_q_r=True, q_max_base=None, R_0_base=None, D_trainable=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(batch_input_shape=batch_input_shape))
    if mlp:
        batch_size = batch_input_shape[0]
        if share_q_r:
            batch_size = 1
        model.add(tf.keras.layers.RNN(BatteryRNNCellMLP(dtype=dtype, dt=dt, mlp_trainable=mlp_trainable, batch_size=batch_size, q_max_base=q_max_base, R_0_base=R_0_base, D_trainable=D_trainable), return_sequences=return_sequences, stateful=stateful,dtype=dtype))
    else:
        model.add(tf.keras.layers.RNN(BatteryRNNCell(dtype=dtype, dt=dt), return_sequences=return_sequences, stateful=stateful,dtype=dtype))

    return model

if __name__ == "__main__":
    # test keral model pred

    inputs = np.ones((700,1000), dtype=DTYPE) * np.linspace(1.0,2.0,1000)  # constant load
    inputs = inputs.T[:,:,np.newaxis]

    model = get_model(batch_input_shape=inputs.shape, dt=10.0, mlp=True)
    model.summary()

    start = time()
    pred = model.predict(inputs)
    duration = time() - start
    print("Inf. time: {:.2f} s - {:.3f} ms/step ".format(duration, duration/inputs.shape[1]*1000))

    cmap = matplotlib.cm.get_cmap('Spectral')

    fig = plt.figure()

    plt.subplot(211)
    for i in range(inputs.shape[0]):
        plt.plot(inputs[i,:,0], color=cmap(i/1000))
    plt.ylabel('I (A)')
    plt.grid()

    plt.subplot(212)
    for i in range(pred.shape[0]):
        plt.plot(pred[i,:], color=cmap(i/1000))
    plt.ylabel('Vm (V)')
    plt.grid()

    plt.xlabel('Time (s)')

    plt.show()
