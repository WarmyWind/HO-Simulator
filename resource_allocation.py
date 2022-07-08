'''
本模块包含资源分配函数:
    equal_RB_allocate

暂时未考虑对不同RB的功率分配
'''

import numpy as np
from info_management import *
from data_factory import search_object_form_list_by_no

def equal_RB_allocate(UE_list, all_UE_list, BS:BS, serving_map:ServingMap, HO_flag=False):
    for _UE in UE_list:
        if _UE.GBR_flag and not HO_flag:
            _needed_nRB = _UE.estimate_needed_nRB_by_SINR()
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


def ICIC_RB_allocate(target_UE_list, all_UE_list, BS:BS, serving_map:ServingMap, HO_flag=False):
    result_flag = True
    for _UE in target_UE_list:
        if _UE.state != 'unserved':
            continue
        if _UE.GBR_flag and not HO_flag:
            _needed_nRB = _UE.estimate_needed_nRB_by_SINR()
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
                find_RB_arr_and_serve(BS, _UE, 'center_first', serving_map, needed_nRB=_needed_nRB)
            else:
                result_flag = False


    return result_flag  # 成功


def ICIC_BS_RB_allocate(UE_list, BS:BS, serving_map:ServingMap):
    '''对BS范围内的UE，按一定顺序重新分配RB'''
    UE_in_range_list = BS.UE_in_range

    edge_UE_no_list = []
    center_UE_no_list = []
    '''先找出边缘和中心UE'''
    for _UE_no in UE_in_range_list:
        if _UE_no == 9:
            probe = _UE_no
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
        if _UE_no == 9:
            probe = _UE_no
        _UE = search_object_form_list_by_no(UE_list, _UE_no)
        if _UE.state == 'served' and _UE.serv_BS != BS.no:
            continue
        if _UE.GBR_flag:
            result_flag = ICIC_RB_allocate([_UE], UE_list, BS, serving_map)
            if result_flag:
                if _UE.posi_type == 'center':
                    center_UE_no_list = center_UE_no_list[center_UE_no_list!=_UE.no]
                elif _UE.posi_type == 'edge':
                    edge_UE_no_list = edge_UE_no_list[edge_UE_no_list!=_UE.no]


    '''先对边缘用户做干扰协调'''
    while len(edge_UE_no_list) != 0:
        _UE_no = edge_UE_no_list[0]
        # if not BS.if_RB_full_load(BS.RB_per_UE, max_UE_per_RB=BS.opt_UE_per_RB, RB_type='edge'):
        #     _UE = search_object_form_list_by_no(UE_list, _UE_no)
        #     find_RB_arr_and_serve(BS, _UE, 'edge', serving_map)
        #     edge_UE_no_list = edge_UE_no_list[1:]
        _UE = search_object_form_list_by_no(UE_list, _UE_no)
        result_flag = ICIC_RB_allocate([_UE], UE_list, BS, serving_map)
        if result_flag:
            edge_UE_no_list = edge_UE_no_list[1:]
        else:
            break

    '''再对中心用户分配RB'''
    while len(center_UE_no_list) != 0:
        _UE_no = center_UE_no_list[0]
        # if not BS.if_RB_full_load(BS.RB_per_UE, max_UE_per_RB=BS.opt_UE_per_RB, RB_type='center'):
        #     _UE = search_object_form_list_by_no(UE_list, _UE_no)
        #     find_RB_arr_and_serve(BS, _UE, 'center', serving_map)
        #     center_UE_no_list = center_UE_no_list[1:]
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
            # if _UE.posi_type == 'center':
            #     if _BS.if_RB_full_load(_BS.RB_per_UE, max_UE_per_RB=_BS.opt_UE_per_RB, RB_type='center'): continue
            #     # RB_arr = _BS.resource_map.center_RB_sorted_idx[:RB_per_UE].astype(int)
            # elif _UE.posi_type == 'edge':
            #     if _BS.if_RB_full_load(_BS.RB_per_UE, max_UE_per_RB=_BS.opt_UE_per_RB, RB_type='edge'): continue
            #     # RB_arr = _BS.resource_map.edge_RB_sorted_idx[:RB_per_UE].astype(int)
            # else:
            #     raise Exception('Invalid posi type:', _UE.posi_type)

            _BS.unserve_UE(_UE, serving_map)  # 释放原RB资源,并令UE的serv_BS暂时为-1
            _UE.RB_type = None  # 令UE的RB_type暂时为None
            _UE.state = 'unserved'  # 令UE的状态暂时为‘unserved’
            ICIC_RB_allocate([_UE], UE_list, _BS, serving_map)
            continue
            # find_RB_arr_and_serve(_BS, _UE, _UE.posi_type, serving_map)

    # '''针对额外使用RB的用户重新分配RB'''
    # for _BS in BS_list:
    #     for _UE_no in _BS.resource_map.extra_edge_RB_serv_list:
    #         if not _BS.if_RB_full_load(_BS.RB_per_UE, max_UE_per_RB=_BS.opt_UE_per_RB, RB_type='edge'):
    #             _UE = search_object_form_list_by_no(UE_list, _UE_no)
    #             _BS.unserve_UE(_UE, serving_map)  # 释放原RB资源,并令UE的serv_BS暂时为-1
    #             _UE.RB_type = None  # 令UE的RB_type暂时为None
    #             _UE.state = 'unserved'  # 令UE的状态暂时为‘unserved’
    #             find_RB_arr_and_serve(_BS, _UE, 'edge', serving_map)
    #
    #     for _UE_no in _BS.resource_map.extra_center_RB_serv_list:
    #         if not _BS.if_RB_full_load(_BS.RB_per_UE, max_UE_per_RB=_BS.opt_UE_per_RB, RB_type='center'):
    #             _UE = search_object_form_list_by_no(UE_list, _UE_no)
    #             _BS.unserve_UE(_UE, serving_map)  # 释放原RB资源,并令UE的serv_BS暂时为-1
    #             _UE.RB_type = None  # 令UE的RB_type暂时为None
    #             _UE.state = 'unserved'  # 令UE的状态暂时为‘unserved’
    #             find_RB_arr_and_serve(_BS, _UE, 'center', serving_map)

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

def dynamic_nRB_per_UE_and_ICIC(PARAM, BS_list, UE_list, actor=None, sess=None):
    '''ICIC动态调整干扰协调的RB比例'''

    '''统计各小区内有多少UE（包括未服务的）'''
    count_UE_in_range(UE_list, BS_list)


    '''动态确定每个小区里给UE分多少RB'''
    if PARAM.dynamic_nRB_per_UE:
        for _BS in BS_list:
            state = _BS.nUE_in_range / _BS.nRB
            _nRB_per_UE = actor.get_action([[state]], sess)[0][0]
            _nRB_per_UE = round(_nRB_per_UE)
            _BS.RB_per_UE = _nRB_per_UE


    '''确定各个BS的最大边缘UE数'''
    all_RB_resourse = PARAM.Macro.opt_UE_per_RB * PARAM.nRB
    # average_edge_UE = 0
    edge_ratio_list = []
    for _BS in BS_list:
        _UE_num = _BS.nUE_in_range
        max_edge_UE = np.floor(((all_RB_resourse/(1-PARAM.ICIC.allow_drop_rate)/_BS.RB_per_UE)-_UE_num)/(PARAM.ICIC.RB_partition_num-1))
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

    for _BS in BS_list:
        if PARAM.ICIC.flag:
            _edge_RB_per_partition = np.floor(PARAM.nRB * PARAM.ICIC.RB_for_edge_ratio / PARAM.ICIC.RB_partition_num)
            _center_RB_num = PARAM.nRB - PARAM.ICIC.RB_partition_num * _edge_RB_per_partition
            center_RB_idx = np.arange(_center_RB_num)

            _RB_start_idx = _center_RB_num + PARAM.ICIC.ICIC_RB_group_for_BS[_BS.no] * _edge_RB_per_partition
            _RB_end_idx = _RB_start_idx + _edge_RB_per_partition
            edge_RB_idx = np.arange(_RB_start_idx, _RB_end_idx)
        else:
            center_RB_idx = np.array([])
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
        '''根据SINR由小到大对UE进行排序'''
        SINR_list = []
        for _UE in UE_list:
            if not _UE.active:
                _SINR = np.Inf
            else:
                if init_flag or PARAM.ICIC.ideal_RL_state:
                    _SINR = _UE.RL_state.filtered_SINR_dB
                    if _SINR == None:
                        raise Exception('SINR == None!')
                # elif PARAM.ICIC.ideal_RL_state:
                #     _SINR = _UE.RL_state.filtered_SINR_dB
                #     if _SINR == None:
                #         raise Exception('SINR == None!')
                elif not PARAM.ICIC.ideal_RL_state and not PARAM.ICIC.RL_state_pred_flag:
                    if len(_UE.RL_state.SINR_dB_record_all) < PARAM.ICIC.obsolete_time+1:
                        _SINR = _UE.RL_state.SINR_dB_record_all[0]
                    else:
                        _SINR = _UE.RL_state.SINR_dB_record_all[-(PARAM.ICIC.obsolete_time+1)]
                else:
                    if len(_UE.RL_state.pred_SINR_dB)==0:
                        _SINR = _UE.RL_state.filtered_SINR_dB
                    else:
                        _SINR = _UE.RL_state.pred_SINR_dB[0]
            SINR_list.append(_SINR)

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

        # '''check'''
        # for _BS in BS_list:
        #     if len(_BS.center_UE_in_range) + len(_BS.edge_UE_in_range) != _BS.nUE_in_range:
        #         raise Exception('center UE num + edge UE num != nUE in range')



