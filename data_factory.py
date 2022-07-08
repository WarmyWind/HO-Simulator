'''
本模块包含处理各种仿真数据和类型的函数

'''

import sys
import time
import scipy.io as scio
import numpy as np
from para_init import *
from user_mobility import *
from channel_fading import *
from visualization import *
import matplotlib.pyplot as plt


def init_all(PARAM, Macro_Posi, UE_posi, shadowFad_dB):
    '''初始化所有对象'''
    '''创建BS对象，并加入列表'''
    Macro_BS_list = create_Macro_BS_list(PARAM, Macro_Posi)

    '''创建UE对象，并加入列表'''
    UE_list = create_UE_list(PARAM, UE_posi)

    '''确定GBR用户'''
    decide_GBR_UE(PARAM, UE_list)


    '''初始化信道、服务信息'''
    shadow = ShadowMap(shadowFad_dB)
    large_fading = LargeScaleChannelMap(PARAM.Macro.nBS, PARAM.nUE)
    small_fading = SmallScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)
    instant_channel = InstantChannelMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)

    large_h = large_scale_channel(PARAM, Macro_BS_list, UE_list, shadow)
    large_fading.update(large_h)
    small_h = small_scale_fading(PARAM.nUE, len(Macro_BS_list), PARAM.Macro.nNt)
    small_fading.update(small_h)
    instant_channel.calculate_by_fading(large_fading, small_fading)
    serving_map = ServingMap(PARAM.Macro.nBS, PARAM.nUE)

    return Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map


def create_Macro_BS_list(PARAM, Macro_Posi):
    macro_BS_list = []

    for i in range(PARAM.Macro.nBS):
        if PARAM.ICIC.flag:
            _edge_RB_per_partition = np.floor(PARAM.nRB * PARAM.ICIC.RB_for_edge_ratio / PARAM.ICIC.RB_partition_num)
            _center_RB_num = PARAM.nRB - PARAM.ICIC.RB_partition_num * _edge_RB_per_partition
            center_RB_idx = np.arange(_center_RB_num)

            # _RB_start_idx = _center_RB_num + np.mod(i, PARAM.ICIC.RB_partition_num) * _edge_RB_per_partition
            _RB_start_idx = _center_RB_num + PARAM.ICIC.ICIC_RB_group_for_BS[i] * _edge_RB_per_partition
            _RB_end_idx = _RB_start_idx + _edge_RB_per_partition
            edge_RB_idx = np.arange(_RB_start_idx, _RB_end_idx)
        else:
            center_RB_idx = np.array([])
            edge_RB_idx = np.array([])

        macro_BS_list.append(BS(i, 'Macro', PARAM.Macro.nNt, PARAM.nRB, PARAM.Macro.Ptmax,
                                Macro_Posi[i], True, PARAM.RB_per_UE, PARAM.Macro.opt_UE_per_RB, PARAM.Macro.nNt,
                                center_RB_idx, edge_RB_idx))
    return macro_BS_list


def create_UE_list(PARAM, UE_posi):
    UE_list = []
    _temp_nUE_per_type = []
    if isinstance(UE_posi, list):

        for i in range(len(UE_posi)):
            if PARAM.nUE_per_type == 'all':
                _nUE_this_type = UE_posi[i].shape[1]
            else:
                _nUE_this_type = PARAM.nUE_per_type[i]
            _temp_nUE_per_type.append(_nUE_this_type)

            _UE_posi_arr = UE_posi[i]
            for _UE_type_no in range(_nUE_this_type):

                _UE_posi = _UE_posi_arr[0, _UE_type_no]
                if _UE_posi != None:
                    _active = True
                else:
                    _active = False
                UE_list.append(UE(len(UE_list), _UE_type_no, _UE_posi, i, active=_active, record_len=PARAM.AHO.obs_len))
    elif len(UE_posi.shape) == 2:
        if PARAM.nUE == 'all':
            _nUE = UE_posi.shape[1]
        else:
            _nUE = PARAM.nUE
        _temp_nUE_per_type.append(_nUE)
        for _UE_no in range(_nUE):
            _UE_posi = UE_posi[0, _UE_no]
            if _UE_posi != None:
                _active = True
            else:
                _active = False
            UE_list.append(UE(_UE_no, _UE_no, _UE_posi, active=_active))
    elif len(UE_posi.shape) == 3:
        for i in range(PARAM.ntype):
            if PARAM.nUE_per_type == 'all':
                _nUE_this_type = UE_posi[i].shape[1]
            else:
                _nUE_this_type = PARAM.nUE_per_type[i]
            _temp_nUE_per_type.append(_nUE_this_type)

            for _UE_type_no in range(_nUE_this_type):
                _UE_posi = UE_posi[i, 0, _UE_type_no]
                if _UE_posi != None:
                    _active = True
                else:
                    _active = False
                UE_list.append(UE(len(UE_list), _UE_type_no, _UE_posi, i, active=_active))

    for _UE in UE_list:
        if isinstance(UE_posi, list) or len(UE_posi.shape) == 3:
            _future_posi = UE_posi[_UE.type][1: 1 + _UE.record_len, _UE.type_no]
        else:
            _future_posi = UE_posi[_UE.no][1: 1 + _UE.record_len]

        _UE.update_future_posi(_future_posi)

    '''更新PARAM'''
    if PARAM.nUE_per_type == 'all':
        PARAM.nUE_per_type = _temp_nUE_per_type
        PARAM.nUE = len(UE_list)
    return UE_list


def decide_GBR_UE(PARAM, UE_list, seed=0):
    np.random.seed(seed)
    nUE = len(UE_list)
    GBR_idx_arr = np.random.choice(nUE, np.floor(nUE*PARAM.GBR_ratio).astype(int))
    for _GBR_idx in GBR_idx_arr:
        _GBR_UE = UE_list[_GBR_idx]
        _GBR_UE.GBR_flag = True
        _GBR_UE.min_rate = PARAM.min_rate

    return

def search_object_form_list_by_no(object_list, no):
    for _obj in object_list:
        if _obj.no == no:
            return _obj



def get_data_from_mat(filepath, index):
    data = scio.loadmat(filepath)
    data = data.get(index)  # 取出字典里的label

    return data


def count_UE_offline(PARAM, UE_list, SINR_th):
    if PARAM.ICIC.flag:
        center_UE = [[] for _ in range(PARAM.nCell)]
        center_UE_offline = []
        edge_UE = [[] for _ in range(PARAM.nCell)]
        edge_UE_offline = []
        UE_on_edge_RB = [[] for _ in range(PARAM.nCell)]
        for _UE in UE_list:
            if _UE.active:
                # if _UE.posi_type == 'center':
                # if _UE.RL_state.filtered_SINR_dB == None:
                #     print('wrong')
                if _UE.RL_state.filtered_SINR_dB == None:
                    raise Exception('RL_state.filtered_SINR_dB == None')
                if _UE.RL_state.filtered_SINR_dB > SINR_th:
                    if _UE.serv_BS == -1:
                        center_UE_offline.append(_UE.no)
                    else:
                        center_UE[_UE.serv_BS].append(_UE.no)
                        if _UE.RB_type == 'edge':
                            UE_on_edge_RB[_UE.serv_BS].append(_UE.no)

                # elif _UE.posi_type == 'edge':
                else:
                    if _UE.serv_BS == -1:
                        edge_UE_offline.append(_UE.no)
                    else:
                        edge_UE[_UE.serv_BS].append(_UE.no)
                        if _UE.RB_type == 'edge':
                            UE_on_edge_RB[_UE.serv_BS].append(_UE.no)

        return center_UE, center_UE_offline, edge_UE, edge_UE_offline, UE_on_edge_RB
    else:
        active_UE = [[] for _ in range(PARAM.nCell)]
        UE_offline = []
        for _UE in UE_list:
            if _UE.active:
                if _UE.serv_BS == -1:
                    UE_offline.append(_UE.no)
                else:
                    active_UE[_UE.serv_BS].append(_UE.no)

        return active_UE, UE_offline