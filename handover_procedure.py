'''
本模块包含HO相关函数:
    handover_criteria_eval
'''


from info_management import *
from resource_allocation import equal_RB_allocate
import numpy as np


def search_object_form_list_by_no(object_list, no):
    for _obj in object_list:
        if _obj.no == no:
            return _obj


def handover_criteria_eval(PARAMS, UE_list, BS_list, large_fading: LargeScaleFadingMap,
                           HOM, TTT, serving_map:ServingMap, quit_criteria = 'avg', allocate_method=equal_RB_allocate):
    for _UE in UE_list:
        if _UE.active == False: continue  # 如果UE不活动，则跳过
        if _UE.state == 'unserved':  continue  # 如果UE不被服务，则跳过
        if _UE.state == 'served':  # 如果UE正被服务，则进行进入HO检测
            _serv_large_h = large_fading.map[_UE.serv_BS, _UE.no]
            _best_BS = np.argmax(large_fading.map[:, _UE.no])
            _best_large_h = large_fading.map[_best_BS, _UE.no]

            '''若目标BS信道超过服务BS一定阈值HOM，触发hanover条件'''
            if _best_BS != _UE.serv_BS and 10*np.log10(_best_large_h) - 10*np.log10(_serv_large_h) >= HOM:
                _UE.update_state('handovering')
                _UE.HO_state.update_target_BS(_best_BS)
                _UE.HO_state.update_duration(0)
                _UE.HO_state.update_target_h_avg(_best_large_h)
                _UE.HO_state.update_h_before(_serv_large_h)

            continue

        '''若在handovering过程，判断是否退出'''
        if _UE.state == 'handovering':
            _target_large_h = large_fading.map[_UE.HO_state.target_BS, _UE.no]
            _UE.HO_state.update_duration(_UE.HO_state.duration + 1)

            if quit_criteria == 'avg':
                _h = (_UE.HO_state.target_h_avg * _UE.HO_state.duration + _target_large_h) / (_UE.HO_state.duration + 1)
            else:
                _h = _target_large_h

            if 10*np.log10(_h) - 10*np.log10(_UE.HO_state.h_before) < HOM:
                '''目标BS信道低于服务BS阈值HOM，HO退出，并记一次HO失败'''
                # _BS_no = _UE.serv_BS
                _UE.quit_handover(False, 'served')
                # _UE.update_state('served')
                continue

                # '''UE尝试接回原BS'''
                # _BS = search_object_form_list_by_no(BS_list, _BS_no)
                # equal_RB_allocate([_UE], _BS, PARAMS.RB_per_UE, serving_map)

            if _UE.HO_state.duration >= TTT:
                '''UE尝试接入目标BS'''
                _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源

                _target_BS = search_object_form_list_by_no(BS_list, _UE.HO_state.target_BS)
                if allocate_method == equal_RB_allocate:
                    _result = allocate_method([_UE], _target_BS, PARAMS.RB_per_UE, serving_map)
                else:
                    raise Exception("Invalid allocate method!", allocate_method)

                _UE.update_state('handovering')
                if _result == True:
                    _UE.quit_handover(True, 'served')
                    # _UE.update_state('served')
                elif _result == False:
                    '''切换失败则进行记录并接入原BS'''
                    _UE.quit_handover(False, 'unserved')
                    # _UE.update_state('unserved')
                    _ = allocate_method([_UE], _serv_BS, PARAMS.RB_per_UE, serving_map)







