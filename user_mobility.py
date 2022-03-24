'''
本模块包含获得用户移动性的方法:
    get_UE_posi_from_mat
'''


import scipy.io as scio
import pandas as pd


def get_UE_posi_from_mat(filepath, index):
    data = scio.loadmat(filepath)
    posi_data = data.get(index)  # 取出字典里的label

    return posi_data


if __name__ == '__main__':
    '''
    用于测试
    '''
    from simulator import Parameter
    from visualization import plot_UE_trajectory
    from network_deployment import cellStructPPP

    PARAM = Parameter()
    Macro_Posi, Micro_Posi, nMicro = cellStructPPP(PARAM.nCell, PARAM.Dist, PARAM.Micro.nBS_avg)
    filepath = 'Set_UE_posi_60s_250user_1to2_new.mat'
    index = 'Set_UE_posi'
    data = get_UE_posi_from_mat(filepath, index)
    # print(data.shape[-1])  # UE轨迹数
    # print(data[:, 0:2])
    plot_UE_trajectory(Macro_Posi, data[:, 214:215])
