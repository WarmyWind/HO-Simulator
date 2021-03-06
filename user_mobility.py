'''
本模块包含关于用户移动性的函数
'''


import scipy.io as scio
import numpy as np
import pandas as pd
from network_deployment import *

def get_UE_posi_from_file(filepath, index):
    '''
    从mat文件读取数据（不限于位置数据，只要满足mat数据格式都可以利用此函数读取）
    :param filepath:  文件路径
    :param index:  mat数据的头索引
    :return:
    '''
    if isinstance(filepath, str):
        if '.mat' in filepath:
            data = scio.loadmat(filepath)
            posi_data = data.get(index)  # 取出字典里的label
        else:
            posi_data = np.load(filepath, allow_pickle=True)

    elif isinstance(filepath, list):
        posi_data = []
        for _path in filepath:
            if '.mat' in _path:
                data = scio.loadmat(_path)
                posi_data.append(data.get(index))
            else:
                data = np.load(_path, allow_pickle=True)
                posi_data.append(data)
        posi_data = np.array(posi_data)

    else:
        raise Exception('Not supported posi format!')
    return posi_data

def process_posi_data(posi_data, fill_symbol = None, dis_threshold=10):
    '''
    对UE轨迹进行处理
    :param posi_data: UE在每一帧的位置数据
    :param fill_symbol: 用户位置如果为空，那么就用fill_symbol填充
    :param dis_threshold: 如果上一帧的位置与这一帧的位置，相距超过了该参数为，那么就视为出现了一个新用户，旧用户消失。
    :return:
    '''
    new_posi_data = []
    if len(posi_data.shape) == 2:
        for _UE_no in range(posi_data.shape[1]):
            _posi = posi_data[:, _UE_no]
            _shift = posi_data[1:, _UE_no]
            _move_distance = np.square(np.abs(_shift - _posi[:-1]))
            _change_point = np.where(_move_distance > dis_threshold)
            if len(_change_point[0]) == 1:
                _change_point = _change_point[0][0] + 1
                _temp1 = np.array([fill_symbol for _ in range(posi_data.shape[0])])
                _temp2 = np.copy(_temp1)
                _temp1[:_change_point] = _posi[:_change_point]
                _temp2[_change_point:] = _posi[_change_point:]

                new_posi_data.append(_temp1)
                new_posi_data.append(_temp2)
            else:
                new_posi_data.append(_posi)

    elif len(posi_data.shape) == 3:
        new_posi_data = []
        for i in range(posi_data.shape[0]):
            _posi_data = posi_data[i]
            new_posi_data.append(process_posi_data(_posi_data))
        return new_posi_data

    return np.array(new_posi_data).transpose()





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
    # filepath = 'Set_UE_posi_60s_250user_1to2_new1.mat'
    filepath = 'Set_UE_posi_100s_500user_v1.mat'
    # filepath = ['Set_UE_posi_100s_500user_v{}.mat'.format(i+1) for i in range(3)]
    index = 'Set_UE_posi'
    data = get_UE_posi_from_mat(filepath, index)
    # print(data.shape[-1])  # UE轨迹数
    # print(data[:, 0:2])
    plot_UE_trajectory(Macro_Posi, data[:, 100:110])

    # real_posi = np.real(data)
    # distance = np.max(data) - np.min(data)
    # print(distance)