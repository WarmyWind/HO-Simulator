'''
本模块包含接入方法:
    find_and_update_neighbour_BS
    access_init
'''

from info_management import *
from resource_allocation import equal_RB_allocate, ICIC_RB_allocate
import numpy as np


def find_and_update_neighbour_BS(BS_list, UE_list, num_neibour, large_channel: LargeScaleChannelMap,
                                 instant_channel: InstantChannelMap, L3_coe=4, ideal_meassure=True):
    BS_no_list = []
    for _BS in BS_list:
        BS_no_list.append(_BS.no)  # 获得BS序号
    BS_no_list = np.array(BS_no_list)

    large_h = large_channel.map[BS_no_list]  # BS对应的大尺度信道
    for _UE in UE_list:
        _UE_no = _UE.no
        # if _UE_no == 77:
        #     _ = 77
        # Offset = np.ones((1, nBS)) * (PARAMS.Macro.PtmaxdBm - PARAMS.Micro.PtmaxdBm - PARAMS.Micro.ABS)
        # RecivePowerdBm = large_h[:, _UE_no] - Offset
        _h = large_h[:, _UE_no]  # 所有基站到该用户的大尺度信道
        _idx = np.argsort(_h.flatten())[::-1]  # 信道响应由大到小
        _neighbour_idx = BS_no_list[_idx[:num_neibour]]  # 最大的几个基站
        _neighbour_idx_before = _UE.neighbour_BS
        _UE.update_neighbour_BS(_neighbour_idx)

        if ideal_meassure:
            L3_h = []
            for _BS_no in _neighbour_idx:
                L3_h.append(large_channel.map[_BS_no, _UE_no])

        else:
            instant_h = instant_channel.map[:, :, _neighbour_idx, _UE_no]
            instant_h_power = np.square(np.abs(instant_h))
            # instant_h_power_mean = np.mean(instant_h_power, axis=0)
            instant_h_power_mean = np.mean(instant_h_power)
            instant_h_mean = np.sqrt(instant_h_power_mean)

            k = (1 / 2) ** (L3_coe / 4)
            L3_h = []
            for i in range(len(_UE.neighbour_BS)):
                n_idx = _UE.neighbour_BS[i]
                if n_idx not in _neighbour_idx_before:
                    L3_h.append(instant_h_mean[i])
                else:
                    _neignour_BS_L3_h_arr = np.array(_UE.neighbour_BS_L3_h)
                    _L3_h_before = _neignour_BS_L3_h_arr[np.where(_neighbour_idx_before == n_idx)][0]
                    L3_h.append((1 - k) * _L3_h_before + k * instant_h_mean[i])

        _UE.update_neighbour_BS_L3_h(L3_h)


def access_init(PARAMS, BS_list, UE_list, instant_channel: InstantChannelMap,
                serving_map: ServingMap):
    '''
    接入初始化
    :param PARAMS: 仿真参数
    :param BS_list: BS列表，元素是BS类
    :param UE_list: UE列表，元素是UE类
    :param instant_channel: 瞬时信道，InstantChannelMap类
    :param serving_map: ServingMap
    :param allocate_method : 分配方法，默认为equal_RB_allocate
    :return:
    '''
    if PARAMS.ICIC.flag == 1:
        allocate_method = ICIC_RB_allocate
    else:
        allocate_method = equal_RB_allocate
    # BS_no_list = []
    # for _BS in BS_list:
    #     BS_no_list.append(_BS.no)  # 获得BS序号

    instant_h = instant_channel.map
    # nBS = len(BS_no_list)  # 可被接入的BS总数
    # nUE = len(UE_list)

    # 根据邻基站列表接入，可能多个用户接入一个基站
    for _UE in UE_list:
        # if _UE.no == 37:
        #     probe = _UE.no
        if not _UE.active: continue
        if _UE.state != 'unserved': continue
        # _UE_no = _UE.no
        # _h = large_h[:, _UE_no]  # 所有基站到该用户的大尺度信道
        # _idx = np.argsort(_h.flatten())  # 信道响应由小到大
        # NewBS_idx = _idx[-1]
        _idx = _UE.neighbour_BS
        NewBS_idx = _idx[0]  # 接入邻基站

        if BS_list[NewBS_idx].is_full_load(PARAMS.RB_per_UE, PARAMS.ICIC.flag):
            # 判断BS是否达到满载，若达到则掉线
            continue


        if allocate_method == equal_RB_allocate or allocate_method == ICIC_RB_allocate:
            # RB_per_UE = BS_list[NewBS_idx].RB_per_UE
            _allo_result = allocate_method([_UE], UE_list, BS_list[NewBS_idx], serving_map)
            if _allo_result:
                '''更新服务基站的L3测量 以及RL state'''
                _instant_h = instant_h[:, :, NewBS_idx, _UE.no]  # nRB, nNt, nBS, nUE
                _instant_h_power = np.square(np.abs(_instant_h))
                # _instant_h_power_mean = np.mean(_instant_h_power, axis=0)
                _instant_h_power_mean = np.mean(_instant_h_power)
                _UE.update_serv_BS_L3_h(np.sqrt(_instant_h_power_mean))
                # _UE.RL_state.update_active(True)
                _UE.reset_ToS()
        else:
            raise Exception("Invalid allocate method!", allocate_method)

    return True



