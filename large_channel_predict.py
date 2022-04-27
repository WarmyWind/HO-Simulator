'''
    本模块用于预测大尺度信道
'''


import numpy as np
from utils import *
from para_init import Parameter
from channel_fading import get_shadow_from_mat
from network_deployment import road_cell_struct
from user_mobility import *
from simulator import create_Macro_BS_list
PARAM = Parameter()

def get_large_channel(PARAMS, BS_list, UE_posi, shadowFad_dB):
    nBS = len(BS_list)  # 所有基站数
    nDrop = UE_posi.shape[0]
    nUE = UE_posi.shape[1]  # 所有用户数

    BS_posi = []
    for _BS in BS_list:
        BS_posi.append(_BS.posi)
    BS_posi = np.array(BS_posi)  # shape = [nBS]

    x_temp = np.floor(np.ceil(np.real(UE_posi) / 0.5)).astype(int)  # shape = [nDrop, nUE]
    y_temp = np.floor(np.ceil((np.imag(UE_posi) - PARAMS.Dist / 2 / np.sqrt(3)) / 0.5)).astype(int)
    x_temp[x_temp > (shadowFad_dB.shape[2] - 1)] = shadowFad_dB.shape[2] - 1
    y_temp[y_temp > (shadowFad_dB.shape[1] - 1)] = shadowFad_dB.shape[1] - 1

    shadow = np.zeros((nDrop, nUE, nBS))
    for iDrop in range(nDrop):
        for iUE in range(nUE):
            for iBS in range(nBS):
                if -1 < x_temp[iDrop, iUE] < shadowFad_dB.shape[2]:
                    shadow[iDrop, iUE, iBS] = shadowFad_dB[iBS][y_temp[iDrop, iUE], x_temp[iDrop, iUE]]
                else:
                    shadow[iDrop, iUE, iBS] = np.Inf

    UE_posi = np.array([UE_posi for _ in range(nBS)])  # shape = [nBS, nDrop, nUE]
    UE_posi = np.rollaxis(UE_posi, 0, 3)  # shape = [nDrop, nUE, nBS]
    distServer = np.abs(UE_posi - BS_posi)  # shape = [nDrop, nUE, nBS]


    antGain = PARAMS.pathloss.Macro.antGaindB
    dFactor = PARAMS.pathloss.Macro.dFactordB
    pLoss1m = PARAMS.pathloss.Macro.pLoss1mdB

    large_scale_fading_dB = pLoss1m + dFactor * np.log10(distServer) + shadow - antGain
    # for iDrop in range(nDrop):
    #     for iUE in range(nUE):
    #         _UE_posi = UE_posi[iDrop, iUE]
    #         for iBS in range(nBS):
    #             _BS_posi = BS_list[iBS].posi
    #             if BS_list[iBS].type == 'Macro':
    #                 antGain = PARAMS.pathloss.Macro.antGaindB
    #                 dFactor = PARAMS.pathloss.Macro.dFactordB
    #                 pLoss1m = PARAMS.pathloss.Macro.pLoss1mdB
    #                 # shadow  = PARAMS.pathloss.Macro.shadowdB
    #             else:
    #                 antGain = PARAMS.pathloss.Micro.antGaindB
    #                 dFactor = PARAMS.pathloss.Micro.dFactordB
    #                 pLoss1m = PARAMS.pathloss.Micro.pLoss1mdB
    #                 # shadow  = PARAMS.pathloss.Micro.shadowdB
    #
    #
    #             if _UE_posi == None:
    #                 large_scale_fading_dB[iDrop, iBS, iUE] = -np.Inf
    #             else:
    #                 distServer = np.abs(_UE_posi - _BS_posi)  # 用户-基站距离
    #                 '''
    #                 下面的x_temp和y_temp后加的常数与数据有关
    #                 '''
    #                 x_temp = int(np.ceil(np.real(_UE_posi)/0.5))
    #                 y_temp = int(np.ceil((np.imag(_UE_posi)-PARAMS.Dist/2/np.sqrt(3))/0.5))
    #                 x_temp = np.min((shadowFad_dB.shape[2]-1, x_temp))
    #                 y_temp = np.min((shadowFad_dB.shape[1]-1, y_temp))
    #                 shadow = shadowFad_dB[iBS][y_temp, x_temp]
    #                 large_scale_fading_dB[iDrop, iBS, iUE] = pLoss1m + dFactor * np.log10(distServer) + shadow - antGain

    large_scale_channel = 10 ** (-large_scale_fading_dB / 20)
        # print('大尺度衰落(dB)：',large_scale_fading_dB[:,0])
        # large_fading.update(large_scale_fading)
    return large_scale_channel





np.random.seed(0)
'''从文件读取阴影衰落'''
filepath = 'shadowFad_dB_6sigma_60dcov.mat'
index = 'shadowFad_dB'
shadowFad_dB = get_shadow_from_mat(filepath, index)
# probe = shadowFad_dB[0][1]

'''生成BS位置和BS对象列表'''
Macro_Posi = road_cell_struct(PARAM.nCell, PARAM.Dist)
Macro_BS_list = create_Macro_BS_list(PARAM, Macro_Posi)

'''从文件读取UE位置训练集'''
# filepath = 'Set_UE_posi_60s_250user_1to2_new1.mat'

trainset_num_per_type = 200
large_channel = []
UE_posi = []
for i in range(3):
    filepath = 'posi_data/v{}_2000_train.mat'.format(i + 1)
    index = 'Set_UE_posi'
    _UE_posi = get_UE_posi_from_mat(filepath, index)
    _UE_posi = _UE_posi[:, :trainset_num_per_type]
    _UE_posi = process_posi_data(_UE_posi, (1+1j) * np.Inf)
    _large_channel = get_large_channel(PARAM, Macro_BS_list, _UE_posi, shadowFad_dB)

    if large_channel == []:
        large_channel = _large_channel
        UE_posi = _UE_posi
    else:
        large_channel = np.concatenate((large_channel, _large_channel), axis=1)
        UE_posi = np.concatenate((UE_posi, _UE_posi), axis=1)



def create_dataset(large_channel, UE_posi, obs_len=5, pred_len=5):
    x_large_h = []
    x_posi = []
    y_large_h = []

    for iUE in range(large_channel.shape[1]):
        _posi = UE_posi[:, iUE]
        _useful_idx = np.where(_posi != (1+1j)*np.inf)
        _posi = _posi[_useful_idx]
        _large_channel_data = large_channel[_useful_idx,iUE,:]
        for i in np.arange(0,len(_posi)-pred_len,obs_len):
            x_large_h.append(_large_channel_data[i:i+obs_len])
            x_posi.append(_posi[i:i+obs_len])
            y_large_h.append(_large_channel_data[i+obs_len:i+obs_len+pred_len])

    return


train_set = create_dataset(large_channel, UE_posi)