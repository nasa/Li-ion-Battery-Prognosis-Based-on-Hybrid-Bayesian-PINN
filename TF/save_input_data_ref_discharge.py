import numpy as np
import math
from time import time
import argparse
import matplotlib
import matplotlib.pyplot as plt

from battery_data import getDischargeMultipleBatteries

matplotlib.rc('font', size=14)

# load battery data
data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'])
BATTERY_NUM = 3
num_batteries = 8
max_idx_to_use = 100

# data_RW = {BATTERY_NUM: data_RW[BATTERY_NUM]} # only one battery

max_size = np.max([ v[0,0].shape[0] for k,v in data_RW.items() ])

dt = np.diff(data_RW[BATTERY_NUM][2,0])[1]

inputs = None
target = None
inputs_time = None
size_all = []
times_all = []
for k,v in data_RW.items():
    if k>num_batteries:
        continue
    size = []
    times = []
    initial_time = v[2,:][0][0]
    for i,d in enumerate(v[1,:][:max_idx_to_use]):
        size.append(len(d))
        times.append(v[2,:][i][0])
        prep_inputs_time = np.full(max_size, np.nan)
        prep_inp = np.full(max_size, np.nan)
        prep_target = np.full(max_size, np.nan)
        prep_inp[:len(d)] = d
        prep_inputs_time[:len(v[2,:][i])] = v[2,:][i]
        prep_target[:len(v[0,:][i])] = v[0,:][i]
        if inputs is None:
            inputs = prep_inp
            target = prep_target
            inputs_time = prep_inputs_time
        else:
            inputs = np.vstack([inputs, prep_inp])
            target = np.vstack([target, prep_target])
            inputs_time = np.vstack([inputs_time, prep_inputs_time])
    size_all.append(size)
    times_all.append(times)

size_all = np.array(size_all)
times_all = np.array(times_all)

inputs = inputs[:,:,np.newaxis]
time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]

print('inputs.shape:', inputs.shape)

# np.save('./training/input_data_refer_disc_batt_{}.npy'.format(BATTERY_NUM), {'inputs':inputs, 'target':target, 'time':inputs_time})
np.save('./training/input_data_refer_disc_batt_1to8.npy', {'inputs':inputs, 'target':target, 'time':inputs_time, 'sizes': size_all, 'init_time': times_all})
