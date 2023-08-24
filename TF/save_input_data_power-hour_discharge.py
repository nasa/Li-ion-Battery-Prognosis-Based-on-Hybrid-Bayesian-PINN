import numpy as np
import math
from time import time
import argparse
import matplotlib
import matplotlib.pyplot as plt

from battery_data import getDischargeMultipleBatteries

matplotlib.rc('font', size=14)

# load battery data
data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type=None)
BATTERY_NUM = 3
num_batteries = 8
EOD = 3.2

# data_RW = {BATTERY_NUM: data_RW[BATTERY_NUM]} # only one battery

dt = np.diff(data_RW[BATTERY_NUM][2,0])[1]

current_all = []
voltage_all = []
time_all = []
delta_time_all = []
cycles_all = []
power_time_all = []

for k,v in data_RW.items():
    current = []
    voltage = []
    time = []
    delta_time = []
    cycles = []

    if k>num_batteries:
        continue

    initial_time = v[2,:][0][0]
    for i,d in enumerate(v[0,:]):
        current.append(v[1,i][1:])
        voltage.append(v[0,i][1:])
        time.append(v[2,i][1:])
        delta_time.append(np.diff(v[2,i]))

        if len(np.argwhere(v[0,i]<=EOD)):
            cycles.append(v[2,i][np.argwhere(v[0,i]<=EOD)[0][0]])

    delta_time = np.concatenate(delta_time)
    not_zero_dt = delta_time!=0
    current = np.concatenate(current)[not_zero_dt]
    voltage = np.concatenate(voltage)[not_zero_dt]
    time = np.concatenate(time)[not_zero_dt]
    cycles = np.array(cycles)

    power_time = (current * voltage) / delta_time[not_zero_dt]

    current_all.append(current)
    voltage_all.append(voltage)
    time_all.append(time)
    delta_time_all.append(delta_time)
    cycles_all.append(cycles)
    power_time_all.append(power_time)


fig, ax1 = plt.subplots()
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Cumulative Energy (kW/h)')
ax2 = ax1.twinx()
ax2.set_ylabel('Cycles')

for i in range(len(power_time_all)):
    ax1.plot(time_all[i]/3600, np.cumsum(power_time_all[i])/3.6e6, color='C{}'.format(i), label='Batt #{}'.format(i+1))
    ax2.plot(cycles_all[i]/3600, np.arange(cycles_all[i].shape[0])+1, 'x', color='C{}'.format(i))

ax1.grid()
ax1.legend()

fig.savefig('./figures/CumulativeEnergy_Cycles_Batt_1to8.png')


fig = plt.figure()

idx_all = []
for j in range(len(power_time_all)):
    idx = np.array([[i,np.argwhere(time_all[j]==cycles_all[j][i])[0][0]] for i in range(len(cycles_all[j])) if len(np.argwhere(time_all[j]==cycles_all[j][i]))])
    idx_all.append(idx)

    plt.plot(idx[:,0]+1, (np.cumsum(power_time_all[j])/3.6e6)[idx[:,1]], label='Batt #{}'.format(j+1))

plt.ylabel('Cumulative Energy (kW/h)')
plt.xlabel('Cycles')
plt.grid()
plt.legend()

fig.savefig('./figures/CumulativeEnergy_Cycles_vs_CumEnergy_Batt_1to8.png')


plt.show()


# print('power_time.shape:', power_time.shape)

# np.save('./training/input_data_power-hour_batt_{}.npy'.format(BATTERY_NUM), {'power_time':power_time, 'time':time, 'cycles': cycles})
np.save('./training/input_data_power-hour_batt_1to8.npy', {'power_time':power_time_all, 'time':time_all, 'cycles': cycles_all})
