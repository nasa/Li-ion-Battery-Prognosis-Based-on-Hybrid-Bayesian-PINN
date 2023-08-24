import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

DATA_PATH = '../../data/'

BATTERY_FILES = {
    1: 'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    2: 'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    3: 'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    4: 'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    5: 'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    6: 'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    7: 'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    8: 'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    9: 'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat',
    10: 'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat',
    11: 'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat',
    12: 'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat'
}

class BatteryDataFile():
    def __init__(self, mat_file_path):
        mat_contents = loadmat(mat_file_path)

        self.procedure = mat_contents['data'][0,0]['procedure'][0]
        self.description = mat_contents['data'][0,0]['description'][0]

        self.headers = [n[0] for n in mat_contents['data'][0,0]['step'].dtype.descr]

        self.data = mat_contents['data'][0,0]['step'][0,:]
        self.num_steps = len(self.data)

        self.operation_type = np.array([v[0] for v in self.data['type']])

    def getDischarge(self, varnames, min_size=0, discharge_type=None):
        seq_sizes = np.array([len(x[0,:]) for x in self.data[np.where(self.operation_type=='D')[0]][varnames[0]]])

        index = seq_sizes>min_size

        if discharge_type is not None:
            index = index & (self.data[np.where(self.operation_type=='D')[0]]['comment']==discharge_type)

        ret = np.array([
            np.asfarray(x[0,:])
            for x in self.data[np.where(self.operation_type=='D')[0]][varnames[0]][index]
        ])

        for i in np.arange(1,len(varnames)):
            ret = np.vstack([
                ret,
                np.array([
                    np.asfarray(x[0,:])
                    for x in self.data[np.where(self.operation_type=='D')[0]][varnames[i]][index]
                ])
            ])

        return ret

def getDischargeMultipleBatteries(data_path=BATTERY_FILES, varnames=['voltage', 'current', 'relativeTime'], discharge_type='reference discharge'):
    data_dic = {}

    for RWi,path in data_path.items():
        batterty_data = BatteryDataFile(DATA_PATH + data_path[RWi].format(RWi))
        data_dic[RWi] = batterty_data.getDischarge(varnames, discharge_type=discharge_type)
    
    return data_dic

if __name__ == "__main__":
    
    data_RW = getDischargeMultipleBatteries()

    max_idx_to_plot = 1
    
    fig = plt.figure()

    plt.subplot(211)
    for RWi,path in data_RW.items():
        for i,d in enumerate(data_RW[RWi][1,:][:max_idx_to_plot]):
            if i==0:
                plt.plot(data_RW[RWi][2,i], d, label='#RW{}'.format(RWi), color='C{}'.format(RWi))
            else:
                plt.plot(data_RW[RWi][2,i], d, color='C{}'.format(RWi))

    plt.ylabel('Current (A)')
    plt.grid()
    plt.legend()

    plt.subplot(212)
    for RWi,path in data_RW.items():
        for i,d in enumerate(data_RW[RWi][0,:][:max_idx_to_plot]):
            plt.plot(data_RW[RWi][2,i], d, color='C{}'.format(RWi))

    plt.ylabel('Voltage (V)')
    plt.grid()

    plt.xlabel('Time (s)')

    plt.show()
