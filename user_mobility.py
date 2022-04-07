'''
本模块包含获得用户移动性的方法:
    get_UE_posi_from_mat
'''


import scipy.io as scio
import numpy as np
import pandas as pd
from network_deployment import *

def get_UE_posi_from_mat(filepath, index):
    if isinstance(filepath, str):
        data = scio.loadmat(filepath)
        posi_data = data.get(index)  # 取出字典里的label

    elif isinstance(filepath, list):
        posi_data = []
        for _path in filepath:
            data = scio.loadmat(_path)
            posi_data.append(data.get(index))
        posi_data = np.array(posi_data)



    return posi_data


if __name__ == '__main__':
    '''
    用于测试
    '''
    from simulator import Parameter
    from visualization import plot_UE_trajectory
    from network_deployment import cellStructPPP

    PARAM = Parameter()
    # Macro_Posi, Micro_Posi, nMicro = cellStructPPP(PARAM.nCell, PARAM.Dist, PARAM.Micro.nBS_avg)
    Macro_Posi = road_cell_struct(9, 250)
    filepath = 'Set_UE_posi_60s_250user_1to2_new1.mat'
    # filepath = 'Set_UE_posi_100s_500user_v1.mat'
    # filepath = ['Set_UE_posi_100s_500user_v{}.mat'.format(i+1) for i in range(3)]
    index = 'Set_UE_posi'
    data = get_UE_posi_from_mat(filepath, index)
    # print(data.shape[-1])  # UE轨迹数
    # print(data[:, 0:2])
    # plot_UE_trajectory(Macro_Posi, data[2, :, 200:201])

    real_posi = np.real(data)
    distance = np.max(data) - np.min(data)
    print(distance)