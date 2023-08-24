import numpy as np
import math
from time import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from model import get_model

matplotlib.rc('font', size=14)

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)

BATTERY_NUM = 3
q_max_base = 1.0e5
R_0_base = 1.0e1
dt=1.0
checkpoint_filepath = './training/cp_mlp_rw_discharge_batt_{}.ckpt'.format(BATTERY_NUM)
# input_shape = (275, 4301, 1)
input_shape = (823, 4301, 1)
model = get_model(batch_input_shape=input_shape, dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model.load_weights(checkpoint_filepath)
weights = model.get_weights()

data_q_max_R_0_rw = np.load('./training/q_max_R_0_rw_disc_3.npy', allow_pickle=True).item()
data_q_max_R_0_ref = np.load('./training/q_max_R_0_aged_batt_3.npy', allow_pickle=True).item()

time_mean = []
q_max_mean = []
R_0_mean = []
for i in np.arange(2,len(data_q_max_R_0_ref['time']),2):
    idx = (data_q_max_R_0_rw['time']>=data_q_max_R_0_ref['time'][i-2]) & (data_q_max_R_0_rw['time']<=data_q_max_R_0_ref['time'][i+1])
    q_max_mean.append(np.mean(np.hstack([data_q_max_R_0_ref['q_max'][i-2:i+2], data_q_max_R_0_rw['q_max'][idx]])))
    R_0_mean.append(np.mean(np.hstack([data_q_max_R_0_ref['R_0'][i-2:i+2], data_q_max_R_0_rw['R_0'][idx]])))
    time_mean.append(np.mean(data_q_max_R_0_ref['time'][i-2:i+2]))

fig = plt.figure('q_max_over_time')
plt.plot(data_q_max_R_0_rw['time']/3600, data_q_max_R_0_rw['q_max'], '.', label='RW. Disc.')
# plt.plot(data_q_max_R_0_rw['time']/3600, weights[0][::3]*model.layers[0].cell.qMaxBASE.numpy(), '.', label='RW. Disc. new')

plt.plot(data_q_max_R_0_ref['time']/3600, data_q_max_R_0_ref['q_max'], '-o', label='Ref. Disc.')

plt.plot(np.array(time_mean)/3600, q_max_mean, label='Avg.')

plt.xlabel('Time (h)')
plt.ylabel(r'$q_{max}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()
plt.legend()


fig = plt.figure('R_0_over_time')

plt.plot(data_q_max_R_0_rw['time']/3600, data_q_max_R_0_rw['R_0'], '.', label='RW. Disc.')
# plt.plot(data_q_max_R_0_rw['time']/3600, weights[1][::3]*model.layers[0].cell.RoBASE.numpy(), '.', label='RW. Disc. new')

plt.plot(data_q_max_R_0_ref['time']/3600, data_q_max_R_0_ref['R_0'], '-o', label='Ref. Disc.')

plt.plot(np.array(time_mean)/3600, R_0_mean, label='Avg.')

plt.xlabel('Time (h)')
plt.ylabel(r'$R_0$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()
plt.legend()










# predict with avg values

data = np.load('./training/input_data_refer_disc_batt_{}.npy'.format(BATTERY_NUM), allow_pickle=True).item()

inputs_ref = data['inputs']
target_ref = data['target']
inputs_time_ref = data['time']

#get 4 first ref disc
inputs_ref = inputs_ref[:4,:,:]
target_ref = target_ref[:4,:]
inputs_time_ref = inputs_time_ref[:4,:]

time_window_size_ref = inputs_ref.shape[1]
dt = inputs_time_ref[0,1]-inputs_time_ref[0,0]

q_max_base = 1.0e1
R_0_base = 1.0e1
model = get_model(batch_input_shape=(1,time_window_size_ref,1), dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)

model.layers[0].cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))
MLPn_weigths = np.load('./training/MLPn_best_weights.npy',allow_pickle=True)
model.layers[0].cell.MLPn.set_weights([MLPn_weigths[:1], MLPn_weigths[1]])

weights = model.get_weights()

weights[0] = np.reshape(q_max_mean[0], (1,))
weights[1] = np.reshape(R_0_mean[0], (1,))

pred_ref = model.predict(inputs_ref)[:,:,0]

fig = plt.figure()
for i in range(target_ref.shape[0]):
    plt.plot(target_ref[i,:], '-', color='C{}'.format(i))
plt.plot(pred_ref[i,:], '--k',)
# plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.title('Reference Discharge')
plt.grid()

fig = plt.figure('all_lines')
plt.plot(inputs_time_ref.ravel()/3600, target_ref.ravel(), '-', color='C0', label='Ref. Dis.')
plt.plot(inputs_time_ref.ravel()/3600, pred_ref.ravel(), '--', color='C0')


data = np.load('./training/input_data_rw_disc_batt_{}.npy'.format(BATTERY_NUM), allow_pickle=True).item()

inputs_rw = data['inputs']
target_rw = data['target']
inputs_time_rw = data['time']

#get rw within first selec ref discharge
idx = (inputs_time_rw[:,0]>=inputs_time_ref[0,0]) & (inputs_time_rw[:,0]<=inputs_time_ref[-1,0])

inputs_rw = inputs_rw[idx,:,:]
target_rw = target_rw[idx,:]
inputs_time_rw = inputs_time_rw[idx,:]

time_window_size_rw = inputs_rw.shape[1]
dt = inputs_time_rw[0,1]-inputs_time_rw[0,0]


q_max_base = 1.0e1
R_0_base = 1.0e1
model = get_model(batch_input_shape=(1,time_window_size_rw,1), dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)

model.layers[0].cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))
MLPn_weigths = np.load('./training/MLPn_best_weights.npy',allow_pickle=True)
model.layers[0].cell.MLPn.set_weights([MLPn_weigths[:1], MLPn_weigths[1]])

weights = model.get_weights()

weights[0] = np.reshape(q_max_mean[0], (1,))
weights[1] = np.reshape(R_0_mean[0], (1,))

pred_rw = model.predict(inputs_rw)[:,:,0]

fig = plt.figure()
for i in np.arange(0,target_rw.shape[0],10):
    plt.plot(target_rw[i,:], '-', color='C{}'.format(i//10), label='RW Dis. #{}'.format(i))
    plt.plot(pred_rw[i,:], '--k', color='C{}'.format(i//10))

i=target_rw.shape[0]-1
plt.plot(target_rw[i,:], '-', color='C4', label='RW Dis. #{}'.format(i))
plt.plot(pred_rw[i,:], '--k', color='C4')

plt.grid()
# plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.title('RW Discharge')
plt.legend()

fig = plt.figure('all_lines')
plt.plot(inputs_time_rw.ravel()/3600, target_rw.ravel(), '-', color='C1', label='RW. Dis.')
plt.plot(inputs_time_rw.ravel()/3600, pred_rw.ravel(), '--', color='C1')

plt.grid()
plt.legend()

fig = plt.figure()
x = np.arange(target_ref.shape[1])
plt.plot(x, pred_ref[0,:] / target_ref[0,:], color='C0', label='Ref. Dis.')
x = x + (x[-1]+1)
plt.plot(x, pred_ref[1,:] / target_ref[1,:], color='C0')

for i in range(target_rw.shape[0]):
    x_size = np.sum(np.isnan(target_rw[i,:]))
    if x_size<1:
        continue
    x = (x[-1]+1) + np.arange(x_size)
    plt.plot(x, pred_rw[i,:x_size] / target_rw[i,:x_size], color='C1')

plt.plot(x[-1], pred_rw[-1,-1] / target_rw[-1,-1], color='C1', label='RW. Dis.')

x = np.arange(target_ref.shape[1]) + (x[-1]+1)
plt.plot(x, pred_ref[2,:] / target_ref[2,:], color='C0')
x = np.arange(target_ref.shape[1]) + (x[-1]+1)
plt.plot(x, pred_ref[3,:] / target_ref[3,:], color='C0')
plt.grid()
plt.ylabel('Ratio: Pred V / Actual V')
# plt.xlabel('Time')
plt.legend()

# fig = plt.figure()
# x = np.arange(target_ref.shape[1])
# plt.plot(x, (pred_ref[0,:] / target_ref[0,:]) / inputs_ref[0,:,0], color='C0', label='Ref. Dis.')
# x = x + (x[-1]+1)
# plt.plot(x, (pred_ref[1,:] / target_ref[1,:]) / inputs_ref[1,:,0], color='C0')

# for i in range(target_rw.shape[0]):
#     x_size = np.sum(np.isnan(target_rw[i,:]))
#     if x_size<1:
#         continue
#     x = (x[-1]+1) + np.arange(x_size)
#     plt.plot(x, (pred_rw[i,:x_size] / target_rw[i,:x_size]) / inputs_rw[i,:x_size,0], color='C1')

# plt.plot(x[-1], (pred_rw[-1,-1] / target_rw[-1,-1]) / inputs_rw[-1,-1,0], color='C1', label='RW. Dis.')

# x = np.arange(target_ref.shape[1]) + (x[-1]+1)
# plt.plot(x, (pred_ref[2,:] / target_ref[2,:]) / inputs_ref[2,:,0], color='C0')
# x = np.arange(target_ref.shape[1]) + (x[-1]+1)
# plt.plot(x, (pred_ref[3,:] / target_ref[3,:]) / inputs_ref[3,:,0], color='C0')
# plt.grid()
# plt.ylabel('Ratio: (Pred V / Actual V) / Current')
# # plt.xlabel('Time')
# plt.legend()

plt.show()
