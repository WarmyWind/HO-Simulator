'''
本模块包含资源分配的各种函数
'''

import numpy as np
from info_management import *
from data_factory import search_object_form_list_by_no
from copy import deepcopy

def equal_RB_allocate(UE_list, all_UE_list, BS:BS, serving_map:ServingMap, HO_flag=False):
    for _UE in UE_list:
        if _UE.GBR_flag and not HO_flag:
            _needed_nRB = _UE.estimate_needed_nRB_by_SINR('edge', len(BS.edge_RB_idx), len(BS.center_RB_idx))
            if _needed_nRB > BS.nRB:
                _needed_nRB = BS.nRB
        else:
            _needed_nRB = BS.RB_per_UE
        if BS.if_RB_full_load(_needed_nRB, BS.MaxUE_per_RB): return False  # BS已达到满载
        RB_arr = BS.resource_map.RB_sorted_idx[:_needed_nRB]
        Nt_arr = np.array([])
        for _RB in RB_arr:
            # if BS.resource_map.RB_ocp_num[_RB] >= BS.MaxUE_per_RB:
            #     return False  # BS已达到满载
            _random_Nt_range = BS.resource_map.RB_idle_antenna[_RB]
            _Nt = np.random.choice(_random_Nt_range)
            Nt_arr = np.append(Nt_arr, _Nt)

        BS.serve_UE(_UE, RB_arr, Nt_arr, serving_map)
    return True  # 成功


def get_RB_arr_by_type(BS, RB_type, needen_nRB):
    if RB_type == 'center':
        RB_arr = BS.resource_map.center_RB_sorted_idx[:int(needen_nRB)].astype(int)
    elif RB_type == 'edge':
        RB_arr = BS.resource_map.edge_RB_sorted_idx[:int(needen_nRB)].astype(int)
    else:
        RB_arr = BS.resource_map.RB_sorted_idx[:int(needen_nRB)].astype(int)
    return RB_arr

def find_RB_arr_and_serve(BS, _UE, RB_type, serving_map:ServingMap, needed_nRB=None):
    if needed_nRB == None:
        needed_nRB = BS.RB_per_UE
    if RB_type != 'edge_first' and RB_type != 'center_first':
        RB_arr = get_RB_arr_by_type(BS, RB_type, needed_nRB)
        Nt_arr = np.array([])
        for _RB in RB_arr:
            _random_Nt_range = BS.resource_map.RB_idle_antenna[_RB]
            _Nt = np.random.choice(_random_Nt_range)
            Nt_arr = np.append(Nt_arr, _Nt)

        BS.serve_UE(_UE, RB_arr, Nt_arr, serving_map)
        _UE.RB_type = RB_type  # 更新UE的RB类型

    elif RB_type == 'edge_first':
        idle_opt_edge_nRB = BS.get_not_full_nRB(max_UE_per_RB=BS.opt_UE_per_RB, RB_type='edge')
        idle_opt_center_nRB = BS.get_not_full_nRB(max_UE_per_RB=BS.opt_UE_per_RB, RB_type='center')
        idle_max_edge_nRB = BS.get_not_full_nRB(max_UE_per_RB=BS.MaxUE_per_RB, RB_type='edge')
        idle_max_center_nRB = BS.get_not_full_nRB(max_UE_per_RB=BS.MaxUE_per_RB, RB_type='center')
        if idle_opt_edge_nRB >= needed_nRB:
            RB_arr = get_RB_arr_by_type(BS, 'edge', needed_nRB)
            _UE.RB_type = 'edge'

        elif idle_opt_edge_nRB + idle_opt_center_nRB >= needed_nRB:
            edge_RB_arr = get_RB_arr_by_type(BS, 'edge', idle_opt_edge_nRB)
            center_RB_arr = get_RB_arr_by_type(BS, 'center', needed_nRB - idle_opt_edge_nRB)
            RB_arr = np.concatenate((edge_RB_arr, center_RB_arr))
            _UE.RB_type = 'edge'

        elif idle_max_edge_nRB + idle_opt_center_nRB >= needed_nRB:
            center_RB_arr = get_RB_arr_by_type(BS, 'center', idle_opt_center_nRB)
            edge_RB_arr = get_RB_arr_by_type(BS, 'edge', needed_nRB - idle_opt_center_nRB)
            RB_arr = np.concatenate((edge_RB_arr, center_RB_arr))
            if len(edge_RB_arr) != 0:
                _UE.RB_type = 'edge'
            else:
                _UE.RB_type = 'center'

        elif idle_max_edge_nRB + idle_max_center_nRB >= needed_nRB:
            edge_RB_arr = get_RB_arr_by_type(BS, 'edge', idle_max_edge_nRB)
            center_RB_arr = get_RB_arr_by_type(BS, 'center', needed_nRB - idle_max_edge_nRB)
            RB_arr = np.concatenate((edge_RB_arr, center_RB_arr))
            _UE.RB_type = 'edge'

        else:
            return

        Nt_arr = np.array([])
        for _RB in RB_arr:
            _random_Nt_range = BS.resource_map.RB_idle_antenna[_RB]
            try:
                _Nt = np.random.choice(_random_Nt_range)
            except:
                raise Exception('np.random.choice get wrong!')
            Nt_arr = np.append(Nt_arr, _Nt)

        BS.serve_UE(_UE, RB_arr, Nt_arr, serving_map)


    elif RB_type == 'center_first':
        idle_opt_edge_nRB = BS.get_not_full_nRB(max_UE_per_RB=BS.opt_UE_per_RB, RB_type='edge')
        idle_opt_center_nRB = BS.get_not_full_nRB(max_UE_per_RB=BS.opt_UE_per_RB, RB_type='center')
        idle_max_edge_nRB = BS.get_not_full_nRB(max_UE_per_RB=BS.MaxUE_per_RB, RB_type='edge')
        idle_max_center_nRB = BS.get_not_full_nRB(max_UE_per_RB=BS.MaxUE_per_RB, RB_type='center')
        if idle_opt_center_nRB >= needed_nRB:
            RB_arr = get_RB_arr_by_type(BS, 'center', needed_nRB)
            _UE.RB_type = 'center'

        elif idle_opt_center_nRB + idle_opt_edge_nRB >= needed_nRB:
            center_RB_arr = get_RB_arr_by_type(BS, 'center', idle_opt_center_nRB)
            edge_RB_arr = get_RB_arr_by_type(BS, 'edge', needed_nRB - idle_opt_center_nRB)
            RB_arr = np.concatenate((edge_RB_arr, center_RB_arr))
            _UE.RB_type = 'center'

        elif idle_max_center_nRB + idle_opt_edge_nRB >= needed_nRB:
            edge_RB_arr = get_RB_arr_by_type(BS, 'edge', idle_opt_edge_nRB)
            center_RB_arr = get_RB_arr_by_type(BS, 'center', needed_nRB - idle_opt_edge_nRB)
            RB_arr = np.concatenate((edge_RB_arr, center_RB_arr))
            if len(center_RB_arr) != 0:
                _UE.RB_type = 'center'
            else:
                _UE.RB_type = 'edge'

        elif idle_max_center_nRB + idle_max_edge_nRB >= needed_nRB:
            center_RB_arr = get_RB_arr_by_type(BS, 'center', idle_max_center_nRB)
            edge_RB_arr = get_RB_arr_by_type(BS, 'edge', needed_nRB - idle_max_center_nRB)
            RB_arr = np.concatenate((edge_RB_arr, center_RB_arr))
            _UE.RB_type = 'center'

        else:
            return

        Nt_arr = np.array([])
        for _RB in RB_arr:
            _random_Nt_range = BS.resource_map.RB_idle_antenna[_RB]
            try:
                _Nt = np.random.choice(_random_Nt_range)
            except:
                raise Exception('np.random.choice get wrong!')
            Nt_arr = np.append(Nt_arr, _Nt)

        BS.serve_UE(_UE, RB_arr, Nt_arr, serving_map)


def ICIC_RB_allocate(target_UE_list, all_UE_list, BS:BS, serving_map:ServingMap, est_nRB_flag=True):
    result_flag = True
    for _UE in target_UE_list:
        if _UE.state != 'unserved':
            continue
        if _UE.GBR_flag and est_nRB_flag:
            _needed_nRB = _UE.estimate_needed_nRB_by_SINR('edge', len(BS.edge_RB_idx), len(BS.center_RB_idx))
            _needed_nRB = _needed_nRB + 3
            if _needed_nRB > len(BS.center_RB_idx) + len(BS.edge_RB_idx):
                _needed_nRB = len(BS.center_RB_idx) + len(BS.edge_RB_idx)
        else:
            _needed_nRB = BS.RB_per_UE

        if _UE.posi_type == 'edge':  # 边缘用户
            if not BS.is_full_load(needed_nRB=_needed_nRB, ICIC_flag=True):
                find_RB_arr_and_serve(BS, _UE, 'edge_first', serving_map, needed_nRB=_needed_nRB)
            else:
                result_flag = False

        else:  # 中心用户
            if not BS.is_full_load(needed_nRB=_needed_nRB, ICIC_flag=True):
                # find_RB_arr_and_serve(BS, _UE, 'center_first', serving_map, needed_nRB=_needed_nRB)
                find_RB_arr_and_serve(BS, _UE, 'edge_first', serving_map, needed_nRB=_needed_nRB)
            else:
                result_flag = False


    return result_flag  # 成功


def ICIC_BS_RB_allocate(PARAM, UE_list, BS:BS, serving_map:ServingMap):
    '''对BS范围内的UE，按一定顺序重新分配RB'''
    UE_in_range_list = BS.UE_in_range

    edge_UE_no_list = []
    center_UE_no_list = []
    '''先找出边缘和中心UE'''
    for _UE_no in UE_in_range_list:
        # if not _UE_no in BS.resource_map.serv_UE_list: continue
        _UE = search_object_form_list_by_no(UE_list, _UE_no)
        if _UE.state == 'served' and _UE.serv_BS != BS.no:
            continue
        elif _UE.serv_BS == BS.no:
            BS.unserve_UE(_UE, serving_map)  # 释放原RB资源,并令UE的serv_BS暂时为-1
            _UE.RB_type = None  # 令UE的RB_type暂时为None
            _UE.state = 'unserved'  # 令UE的状态暂时为‘unserved’
        if _UE.posi_type == 'center':
            center_UE_no_list.append(_UE.no)
        elif _UE.posi_type == 'edge':
            edge_UE_no_list.append(_UE.no)

    edge_UE_no_list = np.array(edge_UE_no_list)
    center_UE_no_list = np.array(center_UE_no_list)

    '''优先保证GBR用户'''
    for _UE_no in UE_in_range_list:
        # if _UE_no == 9:
        #     probe = _UE_no
        _UE = search_object_form_list_by_no(UE_list, _UE_no)
        if _UE.state == 'served' and _UE.serv_BS != BS.no:
            continue
        if _UE.GBR_flag:
            result_flag = ICIC_RB_allocate([_UE], UE_list, BS, serving_map, est_nRB_flag=PARAM.ICIC.dynamic)
            if result_flag:
                if _UE.posi_type == 'center':
                    center_UE_no_list = center_UE_no_list[center_UE_no_list!=_UE.no]
                elif _UE.posi_type == 'edge':
                    edge_UE_no_list = edge_UE_no_list[edge_UE_no_list!=_UE.no]


    '''先对边缘用户做干扰协调'''
    while len(edge_UE_no_list) != 0:
        _UE_no = edge_UE_no_list[0]

        _UE = search_object_form_list_by_no(UE_list, _UE_no)
        result_flag = ICIC_RB_allocate([_UE], UE_list, BS, serving_map)
        if result_flag:
            edge_UE_no_list = edge_UE_no_list[1:]
        else:
            break

    '''再对中心用户分配RB'''
    while len(center_UE_no_list) != 0:
        _UE_no = center_UE_no_list[0]

        _UE = search_object_form_list_by_no(UE_list, _UE_no)
        if not BS.is_full_load(needed_nRB=BS.RB_per_UE, ICIC_flag=True):
            try:
                find_RB_arr_and_serve(BS, _UE, 'edge_first', serving_map)
            except:
                raise Exception('Error alloc!')
            center_UE_no_list = center_UE_no_list[1:]
        # result_flag = ICIC_RB_allocate([_UE], UE_list, BS, serving_map)

        else:
            break

    return




def ICIC_RB_reallocate(UE_list, BS_list, serving_map:ServingMap):
    '''针对使用不合法RB的用户和RB类型不一致的用户，重新分配RB'''

    for _UE in UE_list:
        if not _UE.active or _UE.serv_BS == -1: continue

        _BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
        '''针对使用不合法RB的用户，重新分配RB'''
        if _UE.is_in_invalid_RB(_BS, ICIC_flag=True):
            _BS.unserve_UE(_UE, serving_map)  # 释放原RB资源,并令UE的serv_BS暂时为-1
            _UE.RB_type = None  # 令UE的RB_type暂时为None
            _UE.state = 'unserved'  # 令UE的状态暂时为‘unserved’
            ICIC_RB_allocate([_UE], UE_list, _BS, serving_map)
            continue

        '''针对RB类型不一致的用户重新分配RB'''
        if _UE.posi_type == _UE.RB_type: continue
        else:
            _BS.unserve_UE(_UE, serving_map)  # 释放原RB资源,并令UE的serv_BS暂时为-1
            _UE.RB_type = None  # 令UE的RB_type暂时为None
            _UE.state = 'unserved'  # 令UE的状态暂时为‘unserved’
            ICIC_RB_allocate([_UE], UE_list, _BS, serving_map)
            continue
            # find_RB_arr_and_serve(_BS, _UE, _UE.posi_type, serving_map)
    return

# def ICIC_edge_RB_reuse(UE_list, BS:BS, RB_per_UE, serving_map:ServingMap):
#     for _UE in UE_list:
#         if BS.if_RB_full_load(RB_per_UE, 'edge'):
#             return False
#
#         if _UE.RB_type == 'center':
#             BS.unserve_UE(_UE, serving_map)  # 释放原RB资源,并令UE的serv_BS暂时为-1
#             _UE.RB_type = None  # 令UE的RB_type暂时为None
#             _UE.state = 'unserved'  # 令UE的状态暂时为‘unserved’
#
#             RB_arr = BS.resource_map.edge_RB_sorted_idx[:RB_per_UE].astype(int)
#         else:
#             return False
#
#         Nt_arr = np.array([])
#
#
#         for _RB in RB_arr:
#             _random_Nt_range = BS.resource_map.RB_idle_antenna[_RB]
#             _Nt = np.random.choice(_random_Nt_range)
#             Nt_arr = np.append(Nt_arr, _Nt)
#
#         BS.serve_UE(_UE, RB_arr, Nt_arr, serving_map)
#
#         _UE.update_serv_BS(BS.no)
#         _UE.state = 'served'
#         _UE.RB_type = 'edge'  # 更新UE的RB类型
#     return True  # 成功


def count_UE_in_range(UE_list, BS_list):
    '''统计各小区内有多少UE（包括未服务的）'''
    for _BS in BS_list:
        _BS.nUE_in_range = 0
        _BS.UE_in_range = []
    for _UE in UE_list:
        if not _UE.active: continue
        if _UE.serv_BS != -1:
            _BS_no = _UE.serv_BS
        else:
            _BS_no = _UE.neighbour_BS[0]
        _BS = search_object_form_list_by_no(BS_list, _BS_no)
        _BS.nUE_in_range = _BS.nUE_in_range + 1
        _BS.UE_in_range.append(_UE.no)

    '''将UE按SINR排序'''
    try:
        for _BS in BS_list:
            _UE_in_range = _BS.UE_in_range
            _UE_SINR_list = []
            for _UE_no in _UE_in_range:
                _UE = search_object_form_list_by_no(UE_list, _UE_no)
                _UE_SINR_list.append(_UE.RL_state.filtered_SINR_dB)
            arg_idx = np.argsort(_UE_SINR_list)
            _UE_in_range = np.array(_UE_in_range)[arg_idx]
            _BS.UE_in_range = _UE_in_range
    except:
        return


def dynamic_nRB_per_UE_and_ICIC(PARAM, BS_list, UE_list, serving_map:ServingMap, large_fading: LargeScaleChannelMap,
                                instant_channel: InstantChannelMap, extra_interf_map, dynamic_RB_scheme, actor=None, sess=None,
                                 est_by_large_scale=True):
    '''ICIC动态调整干扰协调的RB比例'''
    def handle_ICIC_RB(PARAM, BS_list):
        for _BS in BS_list:
            if PARAM.ICIC.flag and _BS.ICIC_group != -1:
                _edge_RB_per_partition = np.floor(PARAM.nRB * PARAM.ICIC.RB_for_edge_ratio / PARAM.ICIC.RB_partition_num)
                _center_RB_num = PARAM.nRB - PARAM.ICIC.RB_partition_num * _edge_RB_per_partition
                center_RB_idx = np.arange(_center_RB_num)

                _RB_start_idx = _center_RB_num + PARAM.ICIC.ICIC_RB_group_for_BS[_BS.no] * _edge_RB_per_partition
                _RB_end_idx = _RB_start_idx + _edge_RB_per_partition
                edge_RB_idx = np.arange(_RB_start_idx, _RB_end_idx)
            else:
                # center_RB_idx = np.array([])
                center_RB_idx = np.array([RB_idx for RB_idx in range(PARAM.nRB)])
                edge_RB_idx = np.array([])

            _BS.center_RB_idx = center_RB_idx
            _BS.edge_RB_idx = edge_RB_idx

            '''更新resourse_map'''
            edge_RB_sorted_idx = []
            center_RB_sorted_idx = []
            for _RB_no in _BS.resource_map.RB_sorted_idx:
                if _RB_no in center_RB_idx:
                    center_RB_sorted_idx.append(_RB_no)
                elif _RB_no in edge_RB_idx:
                    edge_RB_sorted_idx.append(_RB_no)

            _BS.resource_map.center_RB_sorted_idx = np.array(center_RB_sorted_idx)
            _BS.resource_map.edge_RB_sorted_idx = np.array(edge_RB_sorted_idx)


    '''统计各小区内有多少UE（包括未服务的）'''
    count_UE_in_range(UE_list, BS_list)

    if dynamic_RB_scheme == 'equal':
        if PARAM.dynamic_nRB_per_UE:
            _ICIC_nRB = round(PARAM.ICIC.RB_for_edge_ratio * PARAM.nRB)
            # temp_nRB_per_UE = []
            for _BS in BS_list:
                RB_resource = (_BS.nRB - _ICIC_nRB + _ICIC_nRB / PARAM.ICIC.RB_partition_num) * _BS.opt_UE_per_RB  # there
                _nRB_per_UE = np.round(RB_resource / _BS.nUE_in_range)

                _BS.RB_per_UE = np.min([len(_BS.center_RB_idx)+len(_BS.edge_RB_idx),_nRB_per_UE])


    elif dynamic_RB_scheme == 'old_model':
        '''动态确定每个小区里给UE分多少RB'''
        if PARAM.dynamic_nRB_per_UE:
            for _BS in BS_list:
                state = _BS.nUE_in_range / _BS.nRB
                _nRB_per_UE = actor.get_action([[state]], sess)[0][0]
                _nRB_per_UE = np.round(_nRB_per_UE)
                _BS.RB_per_UE = np.min([len(_BS.center_RB_idx)+len(_BS.edge_RB_idx),_nRB_per_UE])
        '''确定各个BS的最大边缘UE数'''
        all_RB_resourse = PARAM.Macro.opt_UE_per_RB * PARAM.nRB
        # average_edge_UE = 0
        edge_ratio_list = []
        for _BS in BS_list:
            if _BS.ICIC_group != -1:
                _UE_num = _BS.nUE_in_range
                max_edge_UE = np.floor(
                    ((all_RB_resourse / (1 - PARAM.ICIC.allow_drop_rate) / _BS.RB_per_UE) - _UE_num) / (
                                PARAM.ICIC.RB_partition_num - 1))
                max_edge_UE = np.min([_UE_num, max_edge_UE])
                if max_edge_UE < 0:
                    max_edge_UE = 0
                _BS.max_edge_UE_num = max_edge_UE
                # if _BS.no != BS_list[0].no and _BS.no != BS_list[-1].no:
                #     average_edge_UE = average_edge_UE + max_edge_UE
                # _BS.edge_UE_num = 0
                _edge_nRB = _BS.RB_per_UE * _BS.max_edge_UE_num / PARAM.Macro.opt_UE_per_RB
                _edge_ratio = _edge_nRB * PARAM.ICIC.RB_partition_num / PARAM.nRB
                # if _BS.no != BS_list[0].no and _BS.no != BS_list[-1].no:
                if _edge_ratio > 1:
                    _edge_ratio = 1
                edge_ratio_list.append(_edge_ratio)

        # average_edge_UE = average_edge_UE/(len(BS_list)-2)

        '''更新干扰协调的RB比例'''
        # edge_RB_num = average_edge_UE * _BS.RB_per_UE / PARAM.Macro.opt_UE_per_RB
        # PARAM.ICIC.RB_for_edge_ratio = edge_RB_num * 2 / PARAM.nRB
        PARAM.ICIC.RB_for_edge_ratio = np.mean(edge_ratio_list)
        if PARAM.ICIC.RB_for_edge_ratio > 1:
            PARAM.ICIC.RB_for_edge_ratio = 1

    else:
        '''动态确定每个小区里给UE分多少RB,并遍历最优的正交RB数'''
        if PARAM.dynamic_nRB_per_UE:
            neighbour_nUE_list = [0 for _ in range(len(BS_list))]
            neighbour_ICIC_nUE_list = [0 for _ in range(len(BS_list))]
            nUE = 0
            for _BS in BS_list:
                _neighbour_nUE = 0
                _neighbour_ICIC_nUE = 0
                for _neighbour_BS in BS_list:
                    if _neighbour_BS.no == _BS.no:
                        continue
                    _neighbour_nUE = _neighbour_nUE + _neighbour_BS.nUE_in_range
                    if _neighbour_BS.ICIC_group == _BS.ICIC_group:
                        _neighbour_ICIC_nUE = _neighbour_ICIC_nUE + _neighbour_BS.nUE_in_range
                neighbour_nUE_list[_BS.no] = _neighbour_nUE
                neighbour_ICIC_nUE_list[_BS.no] = _neighbour_ICIC_nUE
                nUE = nUE + _BS.nUE_in_range



            est_sum_rate_record = []
            corresponding_nRB_per_UE = [3 for _ in range(len(BS_list))]
            best_ICIC_nRB = 0
            ICIC_nRB_list = [_nRB * PARAM.ICIC.RB_partition_num for _nRB in range(0, int(np.floor(PARAM.nRB/PARAM.ICIC.RB_partition_num))+1)]

            max_rate_sum = 0
            for ii in range(0, len(ICIC_nRB_list)):
                _ICIC_nRB = ICIC_nRB_list[ii]
                temp_nRB_per_UE=[]
                est_rate_sum = 0
                for _BS in BS_list:
                    if dynamic_RB_scheme == 'new_model':
                        # _ICI = (_BS.nRB - _ICIC_nRB)/_BS.nRB * neighbour_nUE_list[_BS.no] + _ICIC_nRB/_BS.nRB * neighbour_ICIC_nUE_list[_BS.no]
                        # state = [_BS.nUE_in_range / _BS.nRB, _ICI/PARAM.nUE, _ICIC_nRB / _BS.nRB]
                        low_SINR_nUE_in_range = _BS.count_low_SINR_nUE_in_range(UE_list, PARAM.ICIC.SINR_th_for_stat, PARAM.sigma2)
                        state = [nUE/100/_BS.nRB, _BS.nUE_in_range/_BS.nRB, low_SINR_nUE_in_range/_BS.nUE_in_range,  _ICIC_nRB / _BS.nRB]
                        _nRB_per_UE = _BS.nRB * actor.get_action([state], sess)[0][0]
                        _nRB_per_UE = np.round(_nRB_per_UE)
                        temp_nRB_per_UE.append(np.min([len(_BS.center_RB_idx) + len(_BS.edge_RB_idx), _nRB_per_UE]))
                    else:
                        RB_resource = (_BS.nRB - _ICIC_nRB + _ICIC_nRB/PARAM.ICIC.RB_partition_num) * _BS.opt_UE_per_RB  # there
                        _nRB_per_UE = np.round(RB_resource / _BS.nUE_in_range)
                        temp_nRB_per_UE.append(np.min([len(_BS.center_RB_idx) + len(_BS.edge_RB_idx), _nRB_per_UE]))


                temp_instant_channel = deepcopy(instant_channel)
                temp_small_scale_fading = SmallScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.nRB, PARAM.Macro.nNt)
                MCtimes = 8
                for MC_idx in range(MCtimes):
                    # if np.mod(MC_idx,10) == 0:
                    #     print('Current ii:{}, MC time:{}'.format(ii, MC_idx))
                    temp_BS_list = deepcopy(BS_list)
                    temp_UE_list = deepcopy(UE_list)
                    temp_PARAM = deepcopy(PARAM)
                    for _idx in range(len(temp_BS_list)):
                        temp_BS_list[_idx].RB_per_UE = temp_nRB_per_UE[_idx]

                    temp_PARAM.ICIC.RB_for_edge_ratio = _ICIC_nRB / temp_PARAM.nRB
                    handle_ICIC_RB(temp_PARAM, temp_BS_list)
                    ICIC_decide_edge_UE(temp_PARAM, temp_BS_list, temp_UE_list)
                    '''对BS内UE做RB分配'''
                    for _BS in temp_BS_list:
                        ICIC_BS_RB_allocate(temp_PARAM, temp_UE_list, _BS, serving_map)

                    if est_by_large_scale:  # 通过大尺度信道估计系统和数据率
                        '''通过大尺度信道估计系统和数据率'''
                        _sum_rate = 0
                        for _BS in temp_BS_list:
                            _nRB_per_UE = temp_nRB_per_UE[_BS.no]
                            _rate = _BS.estimated_sum_rate(temp_UE_list, temp_PARAM, nRB_per_UE=temp_nRB_per_UE[_BS.no])
                            _sum_rate = _sum_rate + _rate

                        est_rate_sum = _sum_rate
                        break

                    else:  # 通过多次实际测量估计系统和数据率
                        from channel_measurement import update_serv_BS_L3_h
                        '''更新UE的服务基站L3测量'''
                        update_serv_BS_L3_h(temp_UE_list, large_fading, temp_instant_channel, temp_PARAM.L3_coe)

                        '''更新预编码信息和服务记录，10ms更新一次'''
                        from precoding import ZF_precoding
                        for _BS in temp_BS_list:
                            _BS.update_precoding_matrix(temp_instant_channel, ZF_precoding)
                            _BS.serv_UE_list_record.append(_BS.resource_map.serv_UE_list)
                            _BS.RB_ocp_num_record.append(_BS.resource_map.RB_ocp_num)

                        '''统计性能'''
                        from SINR_calculate import get_receive_power, get_interference, calculate_SINR_dB, user_rate
                        rec_P = get_receive_power(temp_BS_list, temp_instant_channel)
                        inter_P = get_interference(temp_PARAM, temp_BS_list, temp_UE_list, temp_instant_channel, extra_interf_map)
                        SINR_dB = calculate_SINR_dB(rec_P, inter_P, temp_PARAM.sigma2)
                        UE_rate = user_rate(temp_PARAM.MLB.RB, SINR_dB, temp_UE_list)
                        _rate_sum = np.sum(UE_rate)
                        est_sum_rate_record.append(_rate_sum)
                        est_rate_sum = np.mean(est_sum_rate_record)

                        '''再次随机化小尺度信道和瞬时信道'''
                        from channel_fading import small_scale_fading
                        small_h = small_scale_fading(len(temp_BS_list), temp_PARAM.nUE, temp_PARAM.nRB, temp_PARAM.Macro.nNt)
                        temp_small_scale_fading.update(small_h)
                        temp_instant_channel.calculate_by_fading(large_fading, temp_small_scale_fading)

                if est_rate_sum > max_rate_sum:
                    max_rate_sum = est_rate_sum
                    corresponding_nRB_per_UE = temp_nRB_per_UE
                    best_ICIC_nRB = _ICIC_nRB

            for BS_idx in range(len(BS_list)):
                BS_list[BS_idx].RB_per_UE = corresponding_nRB_per_UE[BS_idx]

            PARAM.ICIC.RB_for_edge_ratio = best_ICIC_nRB/PARAM.nRB
        else:
            raise Exception('Para Sets fixed nRB for UE!')

    handle_ICIC_RB(PARAM, BS_list)



def ICIC_decide_edge_UE(PARAM, BS_list, UE_list, init_flag = False):
    if not PARAM.ICIC.dynamic:  # 固定边缘RB比例和门限
        for _UE in UE_list:
            if not _UE.active: continue
            if PARAM.ICIC.edge_divide_method == 'SINR':
                _UE.update_posi_type(PARAM.ICIC.SINR_th, PARAM.sigma2)
            else:
                _UE.posi_type = 'center'
                for edge_area_idx in range(PARAM.nCell - 1):
                    if (edge_area_idx + 0.5) * PARAM.Dist - PARAM.ICIC.edge_area_width < np.real(_UE.posi) < (
                            edge_area_idx + 0.5) * PARAM.Dist + PARAM.ICIC.edge_area_width:
                        _UE.posi_type = 'edge'
                        break
    else:
        '''先清空记录'''
        for _BS in BS_list:
            _BS.UE_in_range = []
            _BS.edge_UE_in_range = []
            _BS.center_UE_in_range = []
            # _BS.low_SINR_nUE_in_range = 0
        '''根据SINR由小到大对UE进行排序'''
        SINR_list = []
        for _UE in UE_list:
            if not _UE.active:
                continue
            else:
                if init_flag or PARAM.ICIC.ideal_RL_state:
                    _SINR_dB = _UE.RL_state.filtered_SINR_dB
                    if _SINR_dB == None:
                        raise Exception('SINR == None!')

                elif not PARAM.ICIC.ideal_RL_state and not PARAM.ICIC.RL_state_pred_flag:
                    if len(_UE.RL_state.SINR_dB_record_all) < PARAM.ICIC.obsolete_time+1:
                        _SINR_dB = _UE.RL_state.SINR_dB_record_all[0]
                    else:
                        _SINR_dB = _UE.RL_state.SINR_dB_record_all[-(PARAM.ICIC.obsolete_time+1)]
                else:
                    if len(_UE.RL_state.pred_SINR_dB)==0:
                        _SINR_dB = _UE.RL_state.filtered_SINR_dB
                    else:
                        _SINR_dB = _UE.RL_state.pred_SINR_dB[0]

                # '''根据门限确定SINR低于门限的用户'''
                # if _UE.active:
                #     if _UE.serv_BS != -1:
                #         _BS_no = _UE.serv_BS
                #     else:
                #         _BS_no = _UE.neighbour_BS[0]
                #     _BS = search_object_form_list_by_no(BS_list, _BS_no)
                #     K_raw = np.min([_BS.opt_UE_per_RB, _BS.nUE_in_range])
                #     AG_raw = (_BS.nNt-K_raw+1)/K_raw
                #     K_new = np.min([_BS.MaxUE_per_RB, _BS.nUE_in_range])
                #     AG_new = (_BS.nNt-K_new+1)/K_new
                #     _SINR = 10**(_SINR_dB/10)
                #     _SINR = _SINR*AG_new/AG_raw
                #     _SINR_dB = 10*np.log10(_SINR)
                    # if _SINR_dB < PARAM.ICIC.SINR_th_for_stat:
                    #     _BS.low_SINR_nUE_in_range = _BS.low_SINR_nUE_in_range + 1
                # else:
                #     _BS = BS_list[0]
                #     K_raw = _BS.opt_UE_per_RB
                #     AG_raw = (_BS.nNt - K_raw + 1) / K_raw
                #     K_new = _BS.MaxUE_per_RB
                #     AG_new = (_BS.nNt - K_new + 1) / K_new
                #     _SINR = 10 ** (_SINR / 10)
                #     _SINR = _SINR * AG_new / AG_raw
                #     _SINR = 10 * np.log10(_SINR)

            SINR_list.append(_SINR_dB)


        '''根据排序结果，依次分类是否是边缘用户'''
        sort_idx = np.argsort(SINR_list)
        for _UE_no in sort_idx:
            _UE = UE_list[_UE_no]
            if not _UE.active: continue
            if _UE.serv_BS != -1:
                _BS_no = _UE.serv_BS
            else:
                _BS_no = _UE.neighbour_BS[0]

            _BS = search_object_form_list_by_no(BS_list, _BS_no)
            # if _BS_no == 0:
            #     probe = _BS_no
            _BS.UE_in_range.append(_UE.no)
            if len(_BS.edge_UE_in_range) < _BS.max_edge_UE_num:
                _UE.posi_type = 'edge'
                # _BS.edge_UE_num = _BS.edge_UE_num + 1
                _BS.edge_UE_in_range.append(_UE.no)
            else:
                _UE.posi_type = 'center'
                _BS.center_UE_in_range.append(_UE.no)

        '''check'''
        for _BS in BS_list:
            if len(_BS.center_UE_in_range) + len(_BS.edge_UE_in_range) != _BS.nUE_in_range:
                raise Exception('center UE num + edge UE num != nUE in range')



