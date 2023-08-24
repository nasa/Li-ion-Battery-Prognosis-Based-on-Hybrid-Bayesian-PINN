# %%
from os import path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', size=14)

from aging_model_train import get_model

# load data
# -------------------------------------
data = np.load('./training/input_data_refer_disc_batt_1to8.npy', allow_pickle=True).item()
inputs = data['inputs']
target = data['target']
inputs_time = data['time']

sizes = data['sizes']
init_time = data['init_time']

power_data = np.load('./training/input_data_power-hour_batt_1to8.npy', allow_pickle=True).item()

power_time = power_data['power_time']
time_all = power_data['time']
cycles = power_data['cycles']

data_q_max_R_0 = np.load('./training/q_max_R_0_aged_batt_1to8.npy', allow_pickle=True).item()
q_max_all = data_q_max_R_0['q_max']
R_0_all = data_q_max_R_0['R_0']

batt_index = []
for i,s in enumerate(sizes):
    batt_index += (list(np.ones(len(s), dtype=int)*i))
batt_index = np.array(batt_index)

PWh = [(np.cumsum(power_time[i])/3.6e6) for i in range(len(power_time))]

cum_kWh_ref = []
q_max_ref = []
R_0_ref = []
target_ref = []
time_ref = []
for i in range(len(init_time)):
    cum_kWh_ref.append(
        np.array([PWh[i][np.argwhere(time_all[i]>=init_time[i][j])[0][0]] for j in range(len(init_time[i]))])
    )
    q_max_ref.append(
        q_max_all[batt_index==i]
    )
    R_0_ref.append(
        R_0_all[batt_index==i]
    )
    target_ref.append(
        target[batt_index==i, :]
    )
    time_ref.append(
        inputs_time[batt_index==i, :]
    )

X_all = np.zeros(q_max_all.shape)
for i in range(len(q_max_all)):
    idx = np.argwhere(time_all[batt_index[i]]>=inputs_time[i,0])[0][0]
    X_all[i] = PWh[batt_index[i]][idx]

X_MAX = max(X_all)
Y_MAX_q_max = max(q_max_all)
Y_MAX_R_0 = max(R_0_all)

n_batt = len(q_max_ref)

skip = [3]  # batt 4
# skip = [1] # batt 2
X_test = cum_kWh_ref[skip[0]]
Y_test_q_max = q_max_ref[skip[0]]
Y_test_R_0 = R_0_ref[skip[0]]

idx = [i for i in range(n_batt) if i not in skip]

SAVE_DATA_PATH_q_max = './training/aging_model_v3.npy'
SAVE_DATA_PATH_R_0 = './training/aging_model_v3_R_0.npy'
model_dic_list_q_max = []
if path.exists(SAVE_DATA_PATH_q_max):
    model_dic_list_q_max = np.load(SAVE_DATA_PATH_q_max, allow_pickle=True)

    for batt_i in range(n_batt):
        X = cum_kWh_ref[batt_i]

        model = get_model(batch_size=X.shape[0])
        model.build(input_shape=(X.shape[0],1))
        model.set_weights(model_dic_list_q_max[batt_i]['weights'])
        model_dic_list_q_max[batt_i]['model'] = model

model_dic_list_R_0 = []
if path.exists(SAVE_DATA_PATH_R_0):
    model_dic_list_R_0 = np.load(SAVE_DATA_PATH_R_0, allow_pickle=True)

    for batt_i in range(n_batt):
        X = cum_kWh_ref[batt_i]

        model = get_model(batch_size=X.shape[0])
        model.build(input_shape=(X.shape[0],1))
        model.set_weights(model_dic_list_R_0[batt_i]['weights'])
        model_dic_list_R_0[batt_i]['model'] = model

X_LIM = [0.0, 3.5]
Y_LIM_q_max = [0.4e4, 1.3e4]
Y_LIM_R_0 = [0.5e-1, 3.2e-1]

# %%

if __name__ == "__main__":

    fig, ax1 = plt.subplots()

    for i in range(len(sizes)):
        # if i in batt_skip:
        #     continue
        ax1.plot(np.array(init_time[i])/3600,np.array(sizes[i])/360, 'o', fillstyle='none', color='C{}'.format(i))
    ax1.set_ylabel('Capacity (Ah)')

    ax1.set_xlabel('Time (h)')
    ax1.grid(None)

    ax2 = ax1.twinx()
    for i in range(len(q_max_all)):
        ax2.plot(inputs_time[i,0]/3600, q_max_all[i], '.', color='C{}'.format(batt_index[i]))
    ax2.set_ylabel(r'$q_{MAX}$')
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ax1.plot([], [], 'o', fillstyle='none', color='black', label='Capacity')
    ax1.plot([], [], '.', color='black', label=r'$q_{MAX}$')
    ax1.legend(scatterpoints=1, loc='lower left')

    for i in range(len(sizes)):
        # if i in batt_skip:
        #     continue
        ax2.plot([], [], '.', color='C{}'.format(i), label='Batt #{}'.format(i+1))
    ax2.legend(loc='upper right')


    # %%

    fig, ax1 = plt.subplots()

    for i in range(len(sizes)):
        idx = np.array([np.argwhere(time_all[i]>=init_time[i][j])[0][0] for j in range(len(init_time[i])) if len(np.argwhere(time_all[i]>=init_time[i][j]))])
        ax1.plot(PWh[i][idx],np.array(sizes[i])/360, 'o', fillstyle='none', color='C{}'.format(i))
    ax1.set_ylabel('Capacity (Ah)')

    ax1.set_xlabel('Cumulative Energy (kWh)')
    ax1.grid(None)

    ax2 = ax1.twinx()
    X_all = np.zeros(q_max_all.shape)
    for i in range(len(q_max_all)):
        idx = np.argwhere(time_all[batt_index[i]]>=inputs_time[i,0])[0][0]
        X_all[i] = PWh[batt_index[i]][idx]
        ax2.plot(X_all[i], q_max_all[i], '.', color='C{}'.format(batt_index[i]))
    ax2.set_ylabel(r'$q_{MAX}$')
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ax1.plot([], [], 'o', fillstyle='none', color='black', label='Capacity')
    ax1.plot([], [], '.', color='black', label=r'$q_{MAX}$')
    ax1.legend(scatterpoints=1, loc='lower left')

    for i in range(len(sizes)):
        ax2.plot([], [], '.', color='C{}'.format(i), label='Batt #{}'.format(i+1))
    ax2.legend(loc='upper right')


    # %% R_0
    fig, ax1 = plt.subplots()

    # for i in range(len(sizes)):
    #     # if i in batt_skip:
    #     #     continue
    #     idx = np.array([np.argwhere(time_all[i]>=init_time[i][j])[0][0] for j in range(len(init_time[i])) if len(np.argwhere(time_all[i]>=init_time[i][j]))])
    #     ax1.plot(PWh[i][idx],np.array(sizes[i])/360, 'o', fillstyle='none', color='C{}'.format(i))
    # ax1.set_ylabel('Capacity (Ah)')

    # ax1.set_xlabel('Cumulative Energy (kWh)')
    # ax1.grid(None)

    # ax2 = ax1.twinx()
    ax2 = ax1
    ax2.grid(None)
    for i in range(len(R_0_all)):
        idx = np.argwhere(time_all[batt_index[i]]>=inputs_time[i,0])[0][0]
        ax2.plot(X_all[i], R_0_all[i], '.', color='C{}'.format(batt_index[i]))
    ax2.set_ylabel(r'$R_0$')
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ax1.legend(scatterpoints=1, loc='lower left')

    for i in range(len(sizes)):
        ax2.plot([], [], '.', color='C{}'.format(i), label='Batt #{}'.format(i+1))
    ax2.legend(loc='upper right')


    # %%
    plt.show()
