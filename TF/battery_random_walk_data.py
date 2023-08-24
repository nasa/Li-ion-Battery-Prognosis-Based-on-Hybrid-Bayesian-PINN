# %% imports
import numpy as np
import math
from time import time
import matplotlib
import matplotlib.pyplot as plt

from battery_data import BatteryDataFile, getDischargeMultipleBatteries, DATA_PATH, BATTERY_FILES

matplotlib.rc('font', size=14)
# %% viz batt profile
battery_num = 3
batterty_data = BatteryDataFile(DATA_PATH + BATTERY_FILES[battery_num].format(battery_num))

color_i=0
colors = {}

for i in range(200):
    if batterty_data.data['comment'][i][0] not in colors:
        colors[batterty_data.data['comment'][i][0]] = 'C{}'.format(color_i)
        plt.fill_between(batterty_data.data['time'][i][0,:], batterty_data.data['voltage'][i][0,:], color=colors[batterty_data.data['comment'][i][0]], label=batterty_data.data['comment'][i][0])
        color_i += 1
    else:
        plt.fill_between(batterty_data.data['time'][i][0,:], batterty_data.data['voltage'][i][0,:], color=colors[batterty_data.data['comment'][i][0]])

plt.grid()
# plt.legend(bbox_to_anchor=(1.6,1), borderaxespad=0)
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
# %% load data
# load all battery data
data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type='discharge (random walk)')
# max_idx_to_use = 3
# max_size = np.max([ v[0,0].shape[0] for k,v in data_RW.items() ])
# %% process data
max_size = 0
num_seq = 10

inputs = []
inputs_time = []
target = []

for k,rw_data in data_RW.items():
    # skip batteries RW 9 to 12 for now (RW does not got to EOD)
    if k>8:
        continue
    # rw_data = data_RW[1]
    time = np.hstack([rw_data[2][i] for i in range(len(rw_data[2]))])
    time = time - time[0]
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

time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]
dt = np.diff(data_RW[1][2,0])[1]

# %% plot data
x = [np.arange(len(inputs[i])) for i in range(len(inputs))]
fig = plt.figure()
plt.subplot(311)
for i in range(inputs.shape[0]):
    # plt.fill_between(x[i], inputs[i], alpha=0.4)
    plt.plot(inputs[i,:,0], linewidth=4, alpha=0.8)
plt.ylabel('Current (A)')
plt.grid()

plt.subplot(312)
for i in range(inputs.shape[0]):
    # plt.fill_between(x[i], target[i], 3.2, alpha=0.4)
    plt.plot(target[i], linewidth=4, alpha=0.8)
plt.ylabel('Voltage (V)')
plt.grid()

plt.subplot(313)
for i in range(inputs.shape[0]):
    # plt.fill_between(x[i], target[i]*inputs[i], alpha=0.4)
    plt.plot(target[i]*inputs[i,:,0], linewidth=4, alpha=0.8)
plt.ylabel('Power (W)')
plt.grid()

plt.xlabel('Time (s)')

# %% calc EOD

# move timesteps with earlier EOD
EOD = 3.2

inputs_shiffed = inputs.copy()
target_shiffed = target.copy()
reach_EOD = np.ones(BATCH_SIZE, dtype=int) * time_window_size
for row in np.argwhere((target<EOD) | (np.isnan(target))):
    if reach_EOD[row[0]]>row[1]:
        reach_EOD[row[0]]=row[1]
        row_1 = row[1]
        if ~np.isnan(inputs[row[0],:,0][row[1]]):
            row_1 = row[1] + 1
        inputs_shiffed[row[0],:,0] = np.zeros(time_window_size)
        inputs_shiffed[row[0],:,0][time_window_size-row_1:] = inputs[row[0],:,0][:row_1]
        target_shiffed[row[0]] = np.ones(time_window_size) * target[row[0]][0]
        target_shiffed[row[0]][time_window_size-row_1:] = target[row[0]][:row_1]

# %% plots values slidded in time
inputs_to_plot = inputs_shiffed[::10,:,:]
target_to_plot = target_shiffed[::10,:]
# x = [np.arange(len(inputs_to_plot[i])) for i in range(len(inputs_to_plot))]
fig = plt.figure()
plt.subplot(311)
for i in range(inputs_to_plot.shape[0]):
    # plt.fill_between(x[i], inputs[i], alpha=0.4)
    plt.plot(inputs_to_plot[i,:,0], linewidth=4, alpha=0.8)
plt.ylabel('Current (A)')
plt.grid()

plt.subplot(312)
for i in range(inputs_to_plot.shape[0]):
    # plt.fill_between(x[i], target[i], 3.2, alpha=0.4)
    plt.plot(target_to_plot[i], linewidth=4, alpha=0.8)
plt.ylabel('Voltage (V)')
plt.grid()

plt.subplot(313)
for i in range(inputs_to_plot.shape[0]):
    # plt.fill_between(x[i], target[i]*inputs[i], alpha=0.4)
    plt.plot(target_to_plot[i]*inputs_to_plot[i,:,0], linewidth=4, alpha=0.8)
plt.ylabel('Power (W)')
plt.grid()

plt.xlabel('Time (s)')
# %%
plt.show()
