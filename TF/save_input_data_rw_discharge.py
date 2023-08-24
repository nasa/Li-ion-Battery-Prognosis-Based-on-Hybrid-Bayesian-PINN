import numpy as np
import math
from time import time
import matplotlib
import matplotlib.pyplot as plt

from battery_data import getDischargeMultipleBatteries

matplotlib.rc('font', size=14)

data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type='discharge (random walk)')

BATTERY_NUM = 3
num_batteries = 8
num_seq = 10

# data_RW = {BATTERY_NUM: data_RW[BATTERY_NUM]} # only battery 3

max_size = 0
inputs = []
inputs_time = []
target = []

for k,rw_data in data_RW.items():
    if k>num_batteries:
        continue

    time = np.hstack([rw_data[2][i] for i in range(len(rw_data[2]))])
    # time = time - time[0]
    current_inputs = np.hstack([rw_data[1][i] for i in range(len(rw_data[1]))])
    voltage_target = np.hstack([rw_data[0][i] for i in range(len(rw_data[0]))])

    last_idx = 0
    seq_durations = np.diff([0]+list(np.argwhere(np.diff(time)>10)[:,0]+1))
    
    for curr_duration in seq_durations[:num_seq]:
        if curr_duration>max_size:
            max_size = curr_duration
        curr_idx = last_idx + curr_duration
        inputs.append(current_inputs[last_idx:curr_idx])
        inputs_time.append(time[last_idx:curr_idx])
        target.append(voltage_target[last_idx:curr_idx])
        last_idx = curr_idx

# add nan to end of seq to have all seq in same size
for i in range(len(inputs)):
    prep_inputs = np.full(max_size, np.nan)
    prep_target = np.full(max_size, np.nan)
    prep_inputs_time = np.full(max_size, np.nan)
    prep_inputs[:len(inputs[i])] = inputs[i]
    prep_target[:len(target[i])] = target[i]
    prep_inputs_time[:len(inputs_time[i])] = inputs_time[i]
    inputs[i] = prep_inputs
    target[i] = prep_target
    inputs_time[i] = prep_inputs_time

inputs = np.vstack(inputs)[:,:,np.newaxis]
target = np.vstack(target)
inputs_time = np.vstack(inputs_time)

print('inputs.shape:', inputs.shape)

# # select every other 3 samples
# inputs = inputs[::3,:,:]
# target = target[::3,:]
# inputs_time = inputs_time[::3,:]

# np.save('./training/input_data_rw_disc_batt_{}.npy'.format(BATTERY_NUM), {'inputs':inputs, 'target':target, 'time':inputs_time})
np.save('./training/input_data_rw_disc_batt_1to8.npy', {'inputs':inputs, 'target':target, 'time':inputs_time})
