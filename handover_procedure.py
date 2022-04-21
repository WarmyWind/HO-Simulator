'''
本模块包含HO相关函数:
    handover_criteria_eval
'''

from info_management import *
from resource_allocation import equal_RB_allocate
from radio_access import *
import numpy as np
from utils import *



def handover_criteria_eval(PARAMS, UE_list, BS_list, large_fading: LargeScaleFadingMap,
                           instant_channel: InstantChannelMap,
                           HOM, TTT_list, serving_map: ServingMap, measure_criteria='L3', allocate_method=equal_RB_allocate):
    for _UE in UE_list:
        if _UE.type == 2 and _UE.type_no == 11 and _UE.active == True:
            probe = _UE.type

        if isinstance(TTT_list, list):
            TTT = TTT_list[_UE.type]
        else:
            TTT = TTT_list

        if not _UE.active:
            if _UE.serv_BS != -1:
                _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源
            continue  # 如果UE不活动，则跳过

        if _UE.state == 'unserved':
            continue  # 如果UE不被服务，则跳过
        elif _UE.state == 'served':  # 如果UE正被服务，则进行进入HO检测
            _UE.add_ToS()  # 增加ToS
            if _UE.RL_state.state == 'RLF':  # 在未触发A3之前发生RLF
                _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                _serv_BS.RLF_happen(_UE, serving_map)
                # 尝试接入邻基站
                _ = access_init(PARAMS, BS_list, [_UE], instant_channel, serving_map)

            '''判断最优基站'''
            if measure_criteria == 'avg' or measure_criteria == 'large_h':
                _serv_large_h = np.square(large_fading.map[_UE.serv_BS, _UE.no])
                if _UE.neighbour_BS[0] != _UE.serv_BS:
                    _best_BS = _UE.neighbour_BS[0]
                    _best_large_h = np.square(large_fading.map[_best_BS, _UE.no])
                else:
                    _best_BS = _UE.neighbour_BS[1]
                    _best_large_h = np.square(large_fading.map[_best_BS, _UE.no])

            elif measure_criteria == 'L3':
                _serv_large_h = _UE.serv_BS_L3_h
                if _UE.neighbour_BS[0] != _UE.serv_BS:
                    _best_BS = _UE.neighbour_BS[0]
                    _best_large_h = _UE.neighbour_BS_L3_h[0]
                else:
                    _best_BS = _UE.neighbour_BS[1]
                    _best_large_h = _UE.neighbour_BS_L3_h[1]

            else:
                raise Exception("Invalid measure criteria!", measure_criteria)


            '''若目标BS信道超过服务BS一定阈值HOM，触发hanover条件'''
            if 10 * np.log10(_best_large_h) - 10 * np.log10(_serv_large_h) >= HOM:
                _target_BS = search_object_form_list_by_no(BS_list, _best_BS)
                if not _target_BS.if_full_load():
                    _UE.update_state('handovering')
                    _UE.HO_state.update_target_BS(_best_BS)
                    _UE.HO_state.update_duration(0)
                    _UE.HO_state.update_target_h(_best_large_h)
                    _UE.HO_state.update_h_before(_serv_large_h)

        elif _UE.state == 'handovering':
            '''若在handovering过程，判断是否退出'''
            _UE.add_ToS()
            _UE.HO_state.update_duration(_UE.HO_state.duration + 1)

            if _UE.HO_state.duration < TTT:
                if _UE.RL_state.state == 'RLF':  # 在TTT中发生RLF
                    _UE.quit_handover(False, 'unserved', 0)
                    _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                    _serv_BS.RLF_happen(_UE, serving_map)

                    # 尝试接入邻基站
                    _ = access_init(PARAMS, BS_list, [_UE], instant_channel, serving_map)

                if measure_criteria == 'avg':
                    _target_h = np.square(large_fading.map[_UE.HO_state.target_BS, _UE.no])  # 大尺度信道功率
                    _h = (_UE.HO_state.target_h * _UE.HO_state.duration + _target_h) / (_UE.HO_state.duration + 1)
                    _UE.HO_state.update_target_h(_h)

                elif measure_criteria == 'large_h':
                    _target_large_h = np.square(large_fading.map[_UE.HO_state.target_BS, _UE.no])  # 大尺度信道功率
                    _h = _target_large_h
                    _UE.HO_state.update_target_h(_h)

                elif measure_criteria == 'L3':
                    if _UE.HO_state.target_BS in _UE.neighbour_BS:  # 目标BS在邻小区列表中
                        _neighbour_BS_L3_h = np.array(_UE.neighbour_BS_L3_h)
                        _h = _neighbour_BS_L3_h[np.where(_UE.neighbour_BS == _UE.HO_state.target_BS)]
                        _UE.HO_state.update_target_h(_h)
                    else:  # 目标BS不在邻小区列表中，退出HO且不计HOF
                        _UE.quit_handover(None, 'served')
                        continue

                if 10 * np.log10(_UE.HO_state.target_h) - 10 * np.log10(_UE.HO_state.h_before) < HOM:
                    '''目标BS信道低于服务BS阈值HOM，HO退出，不记HOF'''
                    # _BS_no = _UE.serv_BS
                    _UE.quit_handover(None, 'served')
                    # _UE.update_state('served')
                    continue

                # '''UE尝试接回原BS'''
                # _BS = search_object_form_list_by_no(BS_list, _BS_no)
                # equal_RB_allocate([_UE], _BS, PARAMS.RB_per_UE, serving_map)

            if TTT < _UE.HO_state.duration < TTT + PARAMS.HO_Prep_Time:
                pass

            if _UE.HO_state.duration == TTT + PARAMS.HO_Prep_Time:

                if _UE.RL_state.state == 'out':
                    '''接收HO CMD时信道质量差，记HOF'''
                    _UE.quit_handover(False, 'handovering', 1)
                    _UE.HO_state.HOF_flag = 1
                    continue

                else:
                    '''UE尝试接入目标BS'''
                    _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                    _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源

                    _target_BS = search_object_form_list_by_no(BS_list, _UE.HO_state.target_BS)
                    if allocate_method == equal_RB_allocate:
                        _result = allocate_method([_UE], _target_BS, PARAMS.RB_per_UE, serving_map)
                    else:
                        raise Exception("Invalid allocate method!", allocate_method)

                    _UE.update_state('handovering')

                    if _result:
                        _UE.HO_happen()  # 更新RL state
                    elif not _result:
                        '''不进行记录，接入原BS'''
                        _UE.quit_handover(None, 'unserved')
                        # _UE.update_state('unserved')
                        _ = allocate_method([_UE], _serv_BS, PARAMS.RB_per_UE, serving_map)
                        continue

            if TTT + PARAMS.HO_Prep_Time < _UE.HO_state.duration < TTT + PARAMS.HO_Prep_Time + PARAMS.HO_Exec_Time:

                if len(_UE.RL_state.SINR_record) >= 2 and _UE.RL_state.state == 'out':
                    '''HO 执行时目标BS信道质量差，记HOF'''
                    _UE.quit_handover(False, 'handovering', 2)
                    _UE.HO_state.HOF_flag = 1
                    continue

            if _UE.HO_state.duration == TTT + PARAMS.HO_Prep_Time + PARAMS.HO_Exec_Time:
                if _UE.ToS < _UE.MTS and _UE.HO_state.success_count + _UE.HO_state.failure_count != 0:
                    '''ToS小于最小停留时间(并且有切换记录)，记一次HOF'''
                    _UE.quit_handover(False, 'served', 3)
                    _UE.reset_ToS()
                    continue
                else:
                    '''完成HO，记一次HO成功'''
                    _UE.quit_handover(True, 'served')
                    _UE.reset_ToS()
                    continue
