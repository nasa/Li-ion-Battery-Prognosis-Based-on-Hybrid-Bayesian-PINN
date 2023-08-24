import sys
import numpy as np
import math
import argparse
from time import time
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tensorflow as tf

from model import get_model

# from battery_data import getDischargeMultipleBatteries

matplotlib.rc('font', size=14)

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)

# sys.argv = ['']

parser = argparse.ArgumentParser()
parser.add_argument("--save", default=False, action="store_true" , help="Save plots and results")
args = parser.parse_args()

# %% process data
data_refer = np.load('./training/input_data_refer_disc_batt_1to8.npy', allow_pickle=True).item()
data_rw = np.load('./training/input_data_rw_disc_batt_1to8.npy', allow_pickle=True).item()

inputs_refer = data_refer['inputs']
target_refer = data_refer['target']
inputs_time_refer = data_refer['time']

# subsample timestep to equalize dt - every 10 points
inputs_rw = data_rw['inputs'][:,::10,:]
target_rw = data_rw['target'][:,::10]
inputs_time_rw = data_rw['time'][:,::10]

# %% merge and mix refer and rw
rw_ext_size = inputs_refer.shape[1] - inputs_rw.shape[1]
rw_ext = np.full((inputs_rw.shape[0],rw_ext_size), np.nan)

inputs_rw_ext = np.hstack([inputs_rw[:,:,0],rw_ext])[:,:,np.newaxis]
target_rw_ext = np.hstack([target_rw,rw_ext])
inputs_time_rw_ext = np.hstack([inputs_time_rw,rw_ext])

inputs = np.vstack([inputs_refer, inputs_rw_ext])
target = np.vstack([target_refer, target_rw_ext])
inputs_time = np.vstack([inputs_time_refer, inputs_time_rw_ext])

# %% 

def get_batt_disc_type(i):
    disc_type = 'rw'
    if i<inputs_refer.shape[0]:
        disc_type = 'refer'
        batt_num = i//2
    else:
        i -= inputs_refer.shape[0]
        batt_num = i//10

    return batt_num, disc_type

def get_batt_color(i):
    n,t = get_batt_disc_type(i)
    return 'C{}'.format(n)

def get_batt_marker(i):
    n,t = get_batt_disc_type(i)
    if t=='refer':
        return 'x'
    else:
        return '.'

# %%

time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]
dt = inputs_time[0,3]-inputs_time[0,2]

max_size = inputs.shape[1]
# dt=10.0

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

# %%

# limit to first rw only
inputs_shiffed_all = inputs_shiffed
target_shiffed_all = target_shiffed
inputs_all = inputs
target_all = target
reach_EOD_all = reach_EOD


SIMULATION_OVER_STEPS = 300
inputs_shiffed = np.hstack([inputs_shiffed, (np.ones((BATCH_SIZE, SIMULATION_OVER_STEPS)) * inputs_shiffed[:, -1, 0][:,np.newaxis])[:,:,np.newaxis]])
inputs = np.hstack([inputs, (np.ones((BATCH_SIZE, SIMULATION_OVER_STEPS)) * inputs[:, -1, 0][:,np.newaxis])[:,:,np.newaxis]])
time_window_size = inputs_shiffed.shape[1]

checkpoint_filepath = './training/cp_mlp_MIX_discharge_batt_1to8.ckpt'

q_max_base = 1.0e3
R_0_base = 1.0e1

model_eval = get_model(batch_input_shape=(1,time_window_size-SIMULATION_OVER_STEPS,1), dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model_eval.compile(optimizer='adam', loss="mse", metrics=["mae"])
model = get_model(batch_input_shape=inputs.shape, dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
model.compile(optimizer='adam', loss="mse", metrics=["mae"])

# model.layers[0].cell.MLPp.set_weights(np.load('./training/MLPp_best_weights.npy',allow_pickle=True))
# MLPn_weigths = np.load('./training/MLPn_best_weights.npy',allow_pickle=True)
# model.layers[0].cell.MLPn.set_weights([MLPn_weigths[:1], MLPn_weigths[1]])

xi = np.linspace(0.0,1.0,100)
I = np.ones(100)
fig = plt.figure('MLPp')
# plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]), color='gray')
plt.plot(xi, model.layers[0].cell.MLPp(np.stack([xi,I],1)), color='gray')

fig = plt.figure('MLPn')
# plt.plot(xi, model.layers[0].cell.MLPn(xi[:,np.newaxis]), color='gray')


model.load_weights(checkpoint_filepath)
# model.set_weights(np.load('./training/model_weights_MIX_disc_batt_1to8.npy', allow_pickle=True))
weights = model.get_weights()

# print('U_0p:', weights[2])

if args.save:
    np.save('./training/model_weights_MIX_disc_batt_1to8.npy', weights)


pred_shiffed = model.predict(inputs_shiffed)[:,:,0]

pred = np.full((inputs.shape[0],inputs.shape[1]), np.nan)
for i in range(pred.shape[0]):
    pred[i, :(reach_EOD[i]+SIMULATION_OVER_STEPS)] = pred_shiffed[i, (max_size - reach_EOD[i]):]


# # get mse dist
# mse = np.zeros(inputs.shape[0])
# weights_eval = weights.copy()
# for i in range(inputs.shape[0]):
#     weights_eval[0] = np.reshape(weights[0][i], (1,))
#     weights_eval[1] = np.reshape(weights[1][i], (1,))
#     model_eval.set_weights(weights_eval)
#     mse[i] = model_eval.evaluate(inputs_shiffed[i,:target_shiffed.shape[1],:][np.newaxis,:,:], target_shiffed[i,:][np.newaxis,:,np.newaxis])[0]
#     print("MSE[{}]: {}".format(i, mse[i]))

# print("")
# print("AVG MSE:, ", mse.mean())

# fig = plt.figure()
# plt.hist(mse)
# plt.xlabel(r'mse')
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# if args.save:
#     fig.savefig('./figures/mse_dist_MIX_disc_batt_1to8.png')


fig = plt.figure()
plt.hist(weights[0]*model.layers[0].cell.qMaxBASE.numpy())
plt.xlabel(r'$q_{max}$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

if args.save:
    fig.savefig('./figures/q_max_dist_MIX_disc_batt_1to8.png')

fig = plt.figure('q_max_over_time')
plt.plot(inputs_time[0,0]/3600, weights[0][0]*model.layers[0].cell.qMaxBASE.numpy(), 'xk', label='Refer. Dis.')
plt.plot(inputs_time[-1,0]/3600, weights[0][-1]*model.layers[0].cell.qMaxBASE.numpy(), '.k', label='RW. Dis.')
for i in range(BATCH_SIZE):
    plt.plot(inputs_time[i,0]/3600, weights[0][i]*model.layers[0].cell.qMaxBASE.numpy(), get_batt_marker(i), color=get_batt_color(i))
plt.xlabel('Time (h)')
plt.ylabel(r'$q_{max}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()
plt.legend()

if args.save:
    fig.savefig('./figures/q_max_time_MIX_disc_batt_1to8.png')

fig = plt.figure()
plt.hist(weights[1]*model.layers[0].cell.RoBASE.numpy())
plt.xlabel(r'$R_0$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

if args.save:
    fig.savefig('./figures/R_0_dist_MIX_disc_batt_1to8.png')

fig = plt.figure('R_0_over_time')
plt.plot(inputs_time[0,0]/3600, weights[1][0]*model.layers[0].cell.RoBASE.numpy(), 'xk', label='Refer. Dis.')
plt.plot(inputs_time[-1,0]/3600, weights[1][-1]*model.layers[0].cell.RoBASE.numpy(), '.k', label='RW. Dis.')
for i in range(BATCH_SIZE):
    plt.plot(inputs_time[i,0]/3600, weights[1][i]*model.layers[0].cell.RoBASE.numpy(), get_batt_marker(i), color=get_batt_color(i))
plt.xlabel('Time (h)')
plt.ylabel(r'$R_0$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()
plt.legend()

if args.save:
    fig.savefig('./figures/R_0_time_MIX_disc_batt_1to8.png')

data_to_save = {
    'time': inputs_time[:,0],
    'q_max': weights[0]*model.layers[0].cell.qMaxBASE.numpy(),
    'R_0': weights[1]*model.layers[0].cell.RoBASE.numpy()
}

if args.save:
    np.save('./training/q_max_R_0_MIX_disc_batt_1to8.npy',  data_to_save)

F = 96487.0
V_INT_k = lambda x,i: (2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i)
V_INT = lambda x,A: np.dot(A, np.array([V_INT_k(x,i) for i in range(len(A))])) / F
def Ai(A,i,a):
    A[i]=a
    return A
# reference Redlich-Kister coeff vals
Ap = np.array([-31593.7, 0.106747, 24606.4, -78561.9, 13317.9, 307387.0, 84916.1, -1.07469e+06, 2285.04, 990894.0, 283920, -161513, -469218])

xi = np.linspace(0.0,1.0,100)
Vint = np.array([V_INT(x,Ap) for x in xi])
fig = plt.figure('MLPp')
plt.plot(xi,Vint, label='Redlich V_INT')
# plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]), label='MLP V_INT')
plt.plot(xi, model.layers[0].cell.MLPp(np.stack([xi,I],1)), label='MLP V_INT')
plt.grid()
plt.legend()
if args.save:
    fig.savefig('./figures/MLPp_MIX_disc_batt_1to8.png')

fig = plt.figure('MLPn')
plt.plot(xi, model.layers[0].cell.MLPn(xi[:,np.newaxis]))
plt.grid()
if args.save:
    fig.savefig('./figures/MLPn_MIX_disc_batt_1to8.png')

time_axis = np.arange(time_window_size) * dt
cmap = matplotlib.cm.get_cmap('Spectral')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# I = np.linspace(1.0,4.0,100)
# Z = np.array([model.layers[0].cell.MLPp(np.stack([xi,np.ones(100) * i],1)) for i in I])
# X, Y = np.meshgrid(xi, I)
# ax.contour(X, Y, Z)

for i in np.linspace(1.0,4.0,20):
    I = np.ones(100) * i
    ax.plot(xi, model.layers[0].cell.MLPp(np.stack([xi,I],1)), i, color='C0', zdir='y')



sel_to_plot = [0,3,9,12,20,25,30,35,49]
fig = plt.figure()

plt.subplot(211)

for i in sel_to_plot:
# for i in range(inputs.shape[0]):
    plt.plot(time_axis, inputs[i,:,0], color=get_batt_color(i), alpha=0.5)
plt.ylabel('Current (A)')

plt.grid()
# plt.legend(loc="upper right", ncol=2, bbox_to_anchor=(0,1.02,1,0.2), borderaxespad=0)


plt.subplot(212)
for i in sel_to_plot:
# for i in range(target.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], target[i,:], color=get_batt_color(i), alpha=0.5)
    # plt.plot(time_axis, target[i,:], color='C{}'.format(i), alpha=0.5)
for i in sel_to_plot:
# for i in range(pred.shape[0]):
    idx_end = np.argmax((pred[i,:]<EOD) & (np.isnan(np.concatenate([target[i,:],np.ones(SIMULATION_OVER_STEPS, dtype=bool)]))))
    if idx_end<=0:
        idx_end = len(time_axis)
    plt.plot(time_axis[:idx_end], pred[i,:idx_end], '--', color=get_batt_color(i))
plt.ylabel('Voltage (V)')
plt.grid()

plt.xlabel('Time (s)')

if args.save:
    fig.savefig('./figures/I_V_pred_MIX_disc_batt_1to8.png')

# plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="upper right", mode="expand", borderaxespad=0, ncol=2)


fig = plt.figure()

plt.subplot(211)
for i in range(pred_shiffed.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], inputs[i,:-SIMULATION_OVER_STEPS,0])

plt.ylabel('Current (A)')
plt.grid()

plt.subplot(212)
for i in range(pred.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS],pred[i,:-SIMULATION_OVER_STEPS] / target[i,:])

plt.ylabel('Pred / Actual Ratio (V)')
plt.grid()

plt.xlabel('Time (s)')

if args.save:
    fig.savefig('./figures/pred_ratio_MIX_disc_batt_1to8.png')


reach_EOD_pred = np.ones(inputs.shape[0], dtype=int) * time_window_size
for row in np.argwhere(pred<EOD):
    if reach_EOD_pred[row[0]]>row[1]:
        reach_EOD_pred[row[0]]=row[1]

fig = plt.figure()
EOD_range = [min(np.min(reach_EOD*dt),np.min(reach_EOD_pred*dt)),max(np.max(reach_EOD*dt),np.max(reach_EOD_pred*dt))]
plt.plot(EOD_range, EOD_range, '--k')
plt.plot(reach_EOD*dt, reach_EOD_pred*dt, '.')
plt.ylabel("Predicted EOD (s)")
plt.xlabel("Actual EOD (s)")
plt.xlim(EOD_range)
plt.ylim(EOD_range)
plt.grid()

if args.save:
    fig.savefig('./figures/EOD_pred_MIX_disc_batt_1to8.png')


# %%
batt_test = 8
for batt_test in np.arange(1,9, dtype=int):
    refer_i = int((batt_test-1)*2)
    weights_eval = weights.copy()
    weights_eval[0] = np.reshape(weights[0][refer_i], (1,))
    weights_eval[1] = np.reshape(weights[1][refer_i], (1,))
    model_eval.set_weights(weights_eval)

    pred = model_eval.predict(inputs_refer[refer_i][np.newaxis,:,:])[:,:,0]
    fig = plt.figure()
    xtime = inputs_time_refer[refer_i,:]-inputs_time_refer[refer_i,0]
    plt.plot(xtime, target_refer[refer_i,:], '-k')
    plt.plot(xtime, pred[0,:], '--k')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Battery {} - 1st Refer. Discharge'.format(batt_test))

    if args.save:
        fig.savefig('./figures/pred_MIX_disc_batt_{}_1st-refer.png'.format(batt_test))

    rw_i = int((batt_test-1)*10)
    pred = model_eval.predict(inputs_rw_ext[rw_i][np.newaxis,:,:])[:,:,0]
    fig = plt.figure()
    plt.plot(inputs_time_rw[rw_i,:]-inputs_time_rw[rw_i,0], target_rw[rw_i,:], '-k')
    plt.plot(inputs_time_rw_ext[rw_i,:]-inputs_time_rw_ext[rw_i,0], pred[0,:], '--k')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Battery {} - 1st RW. Discharge'.format(batt_test))

    if args.save:
        fig.savefig('./figures/pred_MIX_disc_batt_{}_1st-rw.png'.format(batt_test))

# %%
plt.show()
