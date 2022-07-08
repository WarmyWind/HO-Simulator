'''
本模块包含HO相关函数:
    handover_criteria_eval
    actice_HO_eval
'''

from info_management import *
from resource_allocation import equal_RB_allocate, ICIC_RB_allocate
from radio_access import *
import numpy as np
from data_factory import search_object_form_list_by_no
from utils import *
from DNN_model_utils import DNN_Model_Wrapper
import torch



def handover_criteria_eval(PARAMS, UE_list, BS_list, large_fading: LargeScaleChannelMap,
                           instant_channel: InstantChannelMap,
                           serving_map: ServingMap, allocate_method, measure_criteria='L3'):
    TTT_list = PARAMS.TTT
    HOM = PARAMS.HOM
    for _UE in UE_list:
        # if _UE.no == 337:
        #     probe = _UE.no
        if isinstance(TTT_list, list):
            TTT = TTT_list[_UE.type]
        else:
            TTT = TTT_list

        if not _UE.active:
            if _UE.serv_BS != -1:
                _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源
            continue  # 如果UE不活动，则跳过
        else:
            if _UE.serv_BS == -1:
                continue  # UE活动但不接入，则跳过

        if _UE.state == 'unserved':
            continue  # 如果UE不被服务，则跳过

        elif _UE.state == 'served' and not _UE.HO_state.handovering:  # 如果UE正被服务，但不在HO，则进行进入HO检测
            _UE.add_ToS()  # 增加ToS
            if _UE.RL_state.state == 'RLF':  # 在未触发A3之前发生RLF
                _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                _serv_BS.RLF_happen(_UE, serving_map)
                # 不尝试接入邻基站
                # _ = access_init(PARAMS, BS_list, [_UE], instant_channel, serving_map)
                continue

            '''判断最优基站'''
            if measure_criteria == 'avg' or measure_criteria == 'large_h':
                _serv_large_h = large_fading.map[_UE.serv_BS, _UE.no]
                if _UE.neighbour_BS[0] != _UE.serv_BS:
                    _best_BS = _UE.neighbour_BS[0]
                    _best_large_h = large_fading.map[_best_BS, _UE.no]
                else:
                    _best_BS = _UE.neighbour_BS[1]
                    _best_large_h = large_fading.map[_best_BS, _UE.no]

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
            if 20 * np.log10(_best_large_h) - 20 * np.log10(_serv_large_h) > HOM:
                # if _UE.no == 337:
                #     print('there')
                _target_BS = search_object_form_list_by_no(BS_list, _best_BS)
                if not _target_BS.is_full_load(PARAMS.RB_per_UE, PARAMS.ICIC.flag):
                    if PARAMS.PHO.ideal_HO:  # 理想切换，哪个基站好就接入哪个
                        '''UE尝试接入目标BS'''
                        _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                        _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源

                        # _target_BS = search_object_form_list_by_no(BS_list, _UE.HO_state.target_BS)
                        if allocate_method == equal_RB_allocate or allocate_method == ICIC_RB_allocate:
                            _result = allocate_method([_UE], UE_list, _target_BS, serving_map, HO_flag=True)
                        else:
                            raise Exception("Invalid allocate method!", allocate_method)
                        if _result:
                            _UE.reset_duration_and_SINR()  # 更新RL state
                        elif not _result:
                            '''不进行记录，不接入原BS'''
                            _UE.quit_handover(None, 'unserved')
                            # _UE.update_state('unserved')
                            # _ = allocate_method([_UE], UE_list, _serv_BS, serving_map)
                            continue
                    else:
                        # _UE.update_state('handovering')
                        _UE.HO_state.handovering = True
                        _UE.HO_state.update_target_BS(_best_BS)
                        _UE.HO_state.update_duration(0)
                        _UE.HO_state.update_target_h(_best_large_h)
                        _UE.HO_state.update_h_before(_serv_large_h)

        # elif _UE.state == 'handovering':
        elif _UE.state == 'served' and _UE.HO_state.handovering:
            '''若在handovering过程，判断是否退出'''
            _UE.add_ToS()
            _UE.HO_state.update_duration(_UE.HO_state.duration + 1)

            if _UE.HO_state.duration < TTT:
                _UE.HO_state.stage = 'TTT'
                if _UE.RL_state.state == 'RLF':  # 在TTT中发生RLF
                    _UE.quit_handover(False, 'unserved', 0)
                    _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                    _serv_BS.RLF_happen(_UE, serving_map)

                    # 不尝试接入邻基站
                    # _ = access_init(PARAMS, BS_list, [_UE], instant_channel, serving_map)
                    continue

                if measure_criteria == 'avg':
                    _target_h_power = np.square(large_fading.map[_UE.HO_state.target_BS, _UE.no])  # 大尺度信道
                    _h_power = (_UE.HO_state.target_h * _UE.HO_state.duration + _target_h_power) / (_UE.HO_state.duration + 1)
                    _UE.HO_state.update_target_h(np.sqrt(_h_power))

                elif measure_criteria == 'large_h':
                    _target_large_h = large_fading.map[_UE.HO_state.target_BS, _UE.no]  # 大尺度信道
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

                if 20 * np.log10(_UE.HO_state.target_h) - 20 * np.log10(_UE.HO_state.h_before) < HOM:
                    '''目标BS信道低于服务BS阈值HOM，HO退出，不记HOF'''
                    # _BS_no = _UE.serv_BS
                    _UE.quit_handover(None, 'served')
                    # _UE.update_state('served')
                    continue

                # '''UE尝试接回原BS'''
                # _BS = search_object_form_list_by_no(BS_list, _BS_no)
                # equal_RB_allocate([_UE], _BS, PARAMS.RB_per_UE, serving_map)

            if TTT < _UE.HO_state.duration < TTT + PARAMS.HO_Prep_Time:
                _UE.HO_state.stage = 'HO_prep'

            if _UE.HO_state.duration == TTT + PARAMS.HO_Prep_Time:

                if _UE.RL_state.state == 'out' and _UE.HO_state.HOF_flag == 0:
                    '''接收HO CMD时信道质量差，记HOF'''
                    _UE.quit_handover(False, 'served', 1)
                    # _UE.HO_state.HOF_flag = 1

                    '''服务基站断开连接'''
                    _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                    _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源
                    _UE.reset_duration_and_SINR()
                    continue

                '''没有发生过晚切换，尝试接入目标基站'''
                _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源
                _target_BS = search_object_form_list_by_no(BS_list, _UE.HO_state.target_BS)
                if allocate_method == equal_RB_allocate or allocate_method == ICIC_RB_allocate:
                    _result = allocate_method([_UE], UE_list, _target_BS, serving_map, HO_flag=True)
                    if _result:
                        # _UE.update_state('served')
                        _UE.reset_duration_and_SINR()  # 更新RL state
                    else:
                        # _UE.update_state('unserved')
                        '''不记录，不接入原BS'''
                        _UE.quit_handover(None, 'unserved')
                        # _UE.update_state('unserved')
                        # _ = allocate_method([_UE], UE_list, _serv_BS, serving_map)
                        continue
                else:
                    raise Exception("Invalid allocate method!", allocate_method)



            if TTT + PARAMS.HO_Prep_Time < _UE.HO_state.duration < TTT + PARAMS.HO_Prep_Time + PARAMS.HO_Exec_Time:
                _UE.HO_state.stage = 'HO_exec'
                if len(_UE.RL_state.SINR_record) >= 2 and _UE.RL_state.state == 'out':
                    '''HO 执行时目标BS信道质量差，记HOF'''
                    _UE.quit_handover(False, 'served', 2)
                    # _UE.HO_state.HOF_flag = 1
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


def actice_HO_eval(PARAMS, NN:DNN_Model_Wrapper, normalize_para, UE_list, BS_list, shadow_map:ShadowMap, large_fading: LargeScaleChannelMap,
                   instant_channel: InstantChannelMap, serving_map: ServingMap, allocate_method, measure_criteria='L3'):
    TTT_list = PARAMS.TTT
    HOM = PARAMS.HOM
    for _UE in UE_list:
        # if _UE.no == 176:
        #     _ = 176

        if isinstance(TTT_list, list):
            TTT = TTT_list[_UE.type]
        else:
            TTT = TTT_list

        if not _UE.active:
            if _UE.serv_BS != -1:
                _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源
            continue  # 如果UE不活动，则跳过
        else:
            if _UE.serv_BS == -1:
                continue  # UE活动但不接入，则跳过

        if _UE.state == 'unserved':
            continue  # 如果UE不被服务，则跳过

        elif _UE.state == 'served' and not _UE.HO_state.handovering:  # 如果UE正被服务但不在HO，则进行进入HO检测
            _UE.add_ToS()  # 增加ToS
            if _UE.RL_state.state == 'RLF':  # 在未触发A3之前发生RLF
                _UE.record_HOF(0)
                _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                _serv_BS.RLF_happen(_UE, serving_map)
                # 不尝试接入邻基站
                # _ = access_init(PARAMS, BS_list, [_UE], instant_channel, serving_map)
                continue

            '''判断最优基站'''
            if measure_criteria == 'avg' or measure_criteria == 'large_h':
                _serv_large_h = large_fading.map[_UE.serv_BS, _UE.no]
                if _UE.neighbour_BS[0] != _UE.serv_BS:
                    _best_BS = _UE.neighbour_BS[0]
                    _best_large_h = large_fading.map[_best_BS, _UE.no]
                else:
                    _best_BS = _UE.neighbour_BS[1]
                    _best_large_h = large_fading.map[_best_BS, _UE.no]

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


            try:
                20 * np.log10(_best_large_h)- 20 * np.log10(_serv_large_h) > HOM
            except:
                raise Exception('Invalid formula!')
            '''若目标BS信道超过服务BS一定阈值HOM，触发hanover条件'''
            if 20 * np.log10(_best_large_h) - 20 * np.log10(_serv_large_h) > HOM:
                # if _UE.no == 77:
                #     _ = 77
                _target_BS = search_object_form_list_by_no(BS_list, _best_BS)

                '''预测大尺度信道'''
                if PARAMS.AHO.ideal_pred:
                    _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                    # if _serv_BS == None:
                    #     print('Wrong')
                    target_h_pred = _UE.cal_future_large_h(PARAMS, _target_BS, shadow_map)
                    serv_h_pred = _UE.cal_future_large_h(PARAMS, _serv_BS, shadow_map)

                    pred_len = np.ceil(TTT / PARAMS.time_resolution).astype(int)
                    target_h_pred = target_h_pred[:pred_len]
                    serv_h_pred = serv_h_pred[:pred_len]

                else:
                    # _posi_record = np.array(_UE.posi_record)
                    # posi_real = (np.real(_posi_record[:, np.newaxis])-normalize_para['mean2'])/normalize_para['sigma2']
                    # posi_imag = (np.real(_posi_record[:, np.newaxis])-normalize_para['mean3'])/normalize_para['sigma3']
                    # x_large_h = (10*np.log10(_UE.all_BS_L3_h_record) - normalize_para['mean1'])/normalize_para['sigma1']
                    # x = np.float32(np.concatenate((x_large_h, posi_real, posi_imag), axis=1))
                    # if PARAMS.AHO.add_noise:
                    #     # x *= (1+PARAMS.AHO.noise * np.random.randn(x.shape[0], x.shape[1]))
                    #     x *= (1+np.random.uniform(-PARAMS.AHO.noise,PARAMS.AHO.noise, size=x.shape))
                    # x = torch.tensor(x)
                    # _pred = np.array(NN.predict(x).detach().cpu())
                    # _pred = _pred.reshape((_UE.record_len, len(BS_list)))
                    # pred_len = np.ceil(TTT/PARAMS.time_resolution).astype(int)
                    # target_h_pred = 10**(_pred[:pred_len, _best_BS]/10)
                    # serv_h_pred = 10**(_pred[:pred_len, _UE.serv_BS]/10)

                    x_large_h = np.float32((10 * np.log10(_UE.all_BS_L3_h_record) - normalize_para['mean1']) / normalize_para[
                        'sigma1'])
                    x_serv = x_large_h[:, _UE.serv_BS]
                    x_target = x_large_h[:, _best_BS]
                    if PARAMS.AHO.add_noise:
                        # x *= (1+PARAMS.AHO.noise * np.random.randn(x.shape[0], x.shape[1]))
                        x_serv *= (1+np.random.uniform(-PARAMS.AHO.noise, PARAMS.AHO.noise, size=x_serv.shape))
                        x_target *= (1 + np.random.uniform(-PARAMS.AHO.noise, PARAMS.AHO.noise, size=x_target.shape))

                    x_serv = torch.tensor(x_serv)
                    serv_h_pred_dB = np.array(NN.predict(x_serv).detach().cpu())
                    serv_h_pred = 10 ** (serv_h_pred_dB / 10)

                    x_target = torch.tensor(x_target)
                    target_h_pred_dB = np.array(NN.predict(x_target).detach().cpu())
                    target_h_pred = 10 ** (target_h_pred_dB / 10)

                    pred_len = np.ceil(TTT / PARAMS.time_resolution).astype(int)
                    serv_h_pred = serv_h_pred.reshape(-1)[:pred_len]
                    target_h_pred = target_h_pred.reshape(-1)[:pred_len]


                pred_meet_result = 20 * np.log10(target_h_pred) - 20 * np.log10(serv_h_pred) >= HOM
                meet_ratio = np.count_nonzero(pred_meet_result) / len(pred_meet_result)

                if meet_ratio >= PARAMS.AHO.pred_allow_ratio:
                    '''主动切换，直接进行HO准备'''
                    if not _target_BS.is_full_load(PARAMS.RB_per_UE, PARAMS.ICIC.flag):
                        # _UE.update_state('handovering')
                        _UE.HO_state.handovering = True
                        _UE.HO_state.update_target_BS(_best_BS)
                        _UE.HO_state.update_duration(0)
                        _UE.HO_state.update_target_h(_best_large_h)
                        _UE.HO_state.update_h_before(_serv_large_h)

        elif _UE.state == 'served' and _UE.HO_state.handovering:
            _UE.add_ToS()
            _UE.HO_state.update_duration(_UE.HO_state.duration + 1)

            if _UE.HO_state.duration < PARAMS.HO_Prep_Time:
                _UE.HO_state.stage = 'HO_prep'

            if _UE.HO_state.duration == PARAMS.HO_Prep_Time:

                if _UE.RL_state.state == 'out' and _UE.HO_state.HOF_flag == 0:
                    '''接收HO CMD时信道质量差，记HOF'''
                    _UE.quit_handover(False, 'served', 1)
                    # _UE.HO_state.HOF_flag = 1

                    '''服务基站断开连接'''
                    _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                    _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源
                    _UE.reset_duration_and_SINR()
                    continue

                '''没有发生过晚切换，尝试接入目标基站'''
                _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源
                _target_BS = search_object_form_list_by_no(BS_list, _UE.HO_state.target_BS)
                if allocate_method == equal_RB_allocate or allocate_method == ICIC_RB_allocate:
                    _result = allocate_method([_UE], UE_list, _target_BS, serving_map, HO_flag=True)
                    if _result:
                        # _UE.update_state('served')
                        _UE.reset_duration_and_SINR()  # 更新RL state
                    else:
                        # _UE.update_state('unserved')
                        '''不记录，不接入原BS'''
                        _UE.quit_handover(None, 'unserved')
                        # _UE.update_state('unserved')
                        # _ = allocate_method([_UE], UE_list, _serv_BS, serving_map)
                        continue
                else:
                    raise Exception("Invalid allocate method!", allocate_method)

            if PARAMS.HO_Prep_Time < _UE.HO_state.duration < PARAMS.HO_Prep_Time + PARAMS.HO_Exec_Time:
                _UE.HO_state.stage = 'HO_exec'
                if len(_UE.RL_state.SINR_record) >= 2 and _UE.RL_state.state == 'out':
                    '''HO 执行时目标BS信道质量差，记HOF'''
                    _UE.quit_handover(False, 'served', 2)
                    _UE.HO_state.HOF_flag = 1
                    continue

            if _UE.HO_state.duration == PARAMS.HO_Prep_Time + PARAMS.HO_Exec_Time:
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
