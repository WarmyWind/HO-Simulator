'''
本模块为仿真器入口
'''



import numpy as np
import time
import os
import copy
from para_init import *
from network_deployment import cellStructPPP, road_cell_struct

from radio_access import access_init, find_and_update_neighbour_BS
from resource_allocation import *
from channel_measurement import *
from SINR_calculate import *
from handover_procedure import *
from utils import *
from data_factory import *
import warnings
warnings.filterwarnings('ignore')

def start_simulation(PARAM, BS_list, UE_list, shadow, large_fading:LargeScaleChannelMap, small_fading:SmallScaleFadingMap,
                     instant_channel:InstantChannelMap, serving_map:ServingMap, NN=None, normalize_para=None):
    '''开始仿真'''

    '''更新UE的邻基站'''
    find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading, instant_channel, PARAM.L3_coe)


    '''初始RL state'''
    update_SS_SINR(UE_list, PARAM.sigma2, PARAM.filter_length_for_SINR)

    '''若考虑干扰协调，划分边缘用户'''
    if PARAM.ICIC.flag:
        if PARAM.ICIC.dynamic:
            ICIC_dynamic_edge_ratio(PARAM, BS_list, UE_list)
        ICIC_decide_edge_UE(PARAM, BS_list, UE_list, init_flag=True)

    '''初始接入'''
    _ = access_init(PARAM, BS_list, UE_list, instant_channel, serving_map)

    '''更新所有基站的L3测量（预测大尺度信道时需要）'''
    if PARAM.active_HO or PARAM.ICIC.RL_state_pred_flag:
        update_all_BS_L3_h_record(UE_list, large_fading, instant_channel, PARAM.L3_coe)

    '''更新UE的服务基站L3测量'''
    update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)


    '''若考虑RB干扰协调，根据UE类型重新分配RB'''
    if PARAM.ICIC.flag:
        for _BS in BS_list:
            ICIC_BS_RB_allocate(UE_list, _BS, PARAM.RB_per_UE, serving_map)


    '''更新预编码信息和服务记录'''
    for _BS in BS_list:
        _BS.update_precoding_matrix(instant_channel, ZF_precoding)
        _BS.serv_UE_list_record.append(_BS.resource_map.serv_UE_list)
        _BS.RB_ocp_num_record.append(_BS.resource_map.RB_ocp_num)

    '''接入时SINR和速率'''
    rec_P = get_receive_power(BS_list, instant_channel)
    inter_P = get_interference(BS_list, UE_list, instant_channel)
    SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma2)
    UE_rate = user_rate(PARAM.MLB.RB, SINR_dB, UE_list)
    # print(np.mean(UE_rate))
    max_inter_P = np.max(inter_P, axis=1)
    max_inter_list = [max_inter_P]
    rate_list = [UE_rate]

    '''统计中心、边缘UE数、掉线率'''
    if PARAM.ICIC.flag:
        center_UE, center_UE_offline, edge_UE, edge_UE_offline, UE_on_edge_RB = count_UE_offline(PARAM, UE_list, SINR_th=PARAM.ICIC.SINR_th_for_stat)
        center_UE_record = [center_UE]
        center_UE_offline_record = [center_UE_offline]
        edge_UE_record = [edge_UE]
        edge_UE_offline_record = [edge_UE_offline]
        UE_on_edge_RB_record = [UE_on_edge_RB]
    else:
        active_UE, UE_offline = count_UE_offline(PARAM, UE_list, SINR_th=PARAM.ICIC.SINR_th_for_stat)
        UE_record = [active_UE]
        UE_offline_record = [UE_offline]

    if PARAM.ICIC.flag and PARAM.ICIC.dynamic:
        RB_for_edge_ratio_list = [PARAM.ICIC.RB_for_edge_ratio]
    else:
        RB_for_edge_ratio_list = []

    '''开始步进时长仿真'''
    for drop_idx in range(1, SimConfig.nDrop):
        # if drop_idx >= 57:
        #     probe = drop_idx
        # if 157 in BS_list[2].resource_map.serv_UE_list:
        #     probe = drop_idx

        for _BS in BS_list:
            flag = np.sum(_BS.resource_map.RB_ocp_num) == len(_BS.resource_map.serv_UE_list) * PARAM.RB_per_UE
            if flag == False:
                print('serv_UE_num * 3 != RB_ocp_num')
                ### 注：此处有小BUG，不知道问题在哪，但不影响运行和总体结果

        '''以下操作均以80ms为步长'''
        if drop_idx % PARAM.posi_resolution == 0:
            '''更新UE位置'''
            _posi_idx = int(drop_idx // PARAM.posi_resolution)
            for _UE in UE_list:
                if isinstance(UE_posi, list):
                    _UE_posi = UE_posi[_UE.type][_posi_idx, _UE.type_no]
                    _UE.update_posi(_UE_posi)
                    _future_posi = UE_posi[_UE.type][_posi_idx+1:_posi_idx+1+_UE.record_len, _UE.type_no]
                    _UE.update_future_posi(_future_posi)
                elif len(UE_posi.shape) == 2:
                    _UE_posi = UE_posi[_posi_idx, _UE.type_no]
                    _UE.update_posi(_UE_posi)
                    _future_posi = UE_posi[_UE.type][_posi_idx + 1:_posi_idx + 1 + _UE.record_len, _UE.type_no]
                    _UE.update_future_posi(_future_posi)
                elif len(UE_posi.shape) == 3:
                    _UE_posi = UE_posi[_UE.type, _posi_idx, _UE.type_no]
                    _UE.update_posi(_UE_posi)
                    _future_posi = UE_posi[_UE.type][_posi_idx + 1:_posi_idx + 1 + _UE.record_len, _UE.type_no]
                    _UE.update_future_posi(_future_posi)

            '''更新小尺度信道信息'''
            small_h = small_scale_fading(PARAM.nUE, len(BS_list), PARAM.Macro.nNt)
            small_fading.update(small_h)

            '''更新大尺度信道信息'''
            if drop_idx % PARAM.posi_resolution == 0:
                large_h = large_scale_channel(PARAM, BS_list, UE_list, shadow)
                large_fading.update(large_h)

            '''更新瞬时信道信息'''
            instant_channel.calculate_by_fading(large_fading, small_fading)

            '''活动UE尝试进行接入，停止活动的UE断开,并初始化RL state'''
            for _UE in UE_list:
                if not _UE.active:
                    if _UE.serv_BS != -1:
                        _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                        _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源
                else:
                    if _UE.serv_BS == -1:
                        _ = access_init(PARAM, BS_list, [_UE], instant_channel, serving_map)
            for _UE in UE_list:
                if not _UE.active or len(_UE.all_BS_L3_h_record) != 0: continue
                update_all_BS_L3_h_record([_UE], large_fading, instant_channel, PARAM.L3_coe)
                find_and_update_neighbour_BS(BS_list, [_UE], PARAM.num_neibour_BS_of_UE, large_fading,
                                             instant_channel, PARAM.L3_coe)
                update_serv_BS_L3_h([_UE], large_fading, instant_channel, PARAM.L3_coe)
                update_SS_SINR([_UE], PARAM.sigma2, PARAM.filter_length_for_SINR)

            '''若假设RL state理想，更新RL state后再ICIC'''
            if PARAM.ICIC.ideal_RL_state:

                '''更新所有基站的L3测量（预测大尺度信道时需要）'''
                if PARAM.active_HO:
                    update_all_BS_L3_h_record(UE_list, large_fading, instant_channel, PARAM.L3_coe)

                '''更新UE的邻基站及其的L3测量'''
                find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading,
                                             instant_channel, PARAM.L3_coe)

                '''更新UE的服务基站L3测量'''
                update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)

                '''更新SS_SINR'''
                update_SS_SINR(UE_list, PARAM.sigma2, PARAM.filter_length_for_SINR)

            else:  # RL state只能用上一帧的值或预测值
                if PARAM.ICIC.RL_state_pred_flag:
                    update_pred_SS_SINR(UE_list, PARAM.sigma2, NN, normalize_para, PARAM.ICIC.RL_state_pred_len)



            '''若考虑干扰协调，划分边缘用户'''
            if PARAM.ICIC.flag:
                if PARAM.ICIC.dynamic and (drop_idx / PARAM.posi_resolution) % PARAM.ICIC.dynamic_period == 0:
                    ICIC_dynamic_edge_ratio(PARAM, BS_list, UE_list)
                ICIC_decide_edge_UE(PARAM, BS_list, UE_list)
                # if not PARAM.ICIC.dynamic:  # 固定边缘RB比例和门限
                #     for _UE in UE_list:
                #         if not _UE.active: continue
                #         if PARAM.ICIC.edge_divide_method == 'SINR':
                #             _UE.update_posi_type(PARAM.ICIC.SINR_th, PARAM.sigma2)
                #         else:
                #             _UE.posi_type = 'center'
                #             for edge_area_idx in range(PARAM.nCell - 1):
                #                 if (edge_area_idx + 0.5) * PARAM.Dist - PARAM.ICIC.edge_area_width < np.real(_UE.posi) < (
                #                         edge_area_idx + 0.5) * PARAM.Dist + PARAM.ICIC.edge_area_width:
                #                     _UE.posi_type = 'edge'
                #                     break
                # else:  # 动态划分RB和边缘用户
                #     ICIC_dynamic_edge_ratio(PARAM, BS_list, UE_list)


            '''若考虑RB干扰协调，根据UE类型分配RB'''
            if PARAM.ICIC.flag:
                '''对BS内UE做RB分配'''
                for _BS in BS_list:
                    # if _BS.no == 2:
                    #     probe = _BS.no
                    ICIC_BS_RB_allocate(UE_list, _BS, PARAM.RB_per_UE, serving_map)

                '''针对额外使用RB的用户和RB类型不一致的用户，重新分配RB'''
                ICIC_RB_reallocate(UE_list, BS_list, PARAM.RB_per_UE, serving_map)

            '''若假设RL state不理想，ICIC后再更新RL state'''
            if not PARAM.ICIC.ideal_RL_state:

                '''更新所有基站的L3测量（预测大尺度信道时需要）'''
                if PARAM.active_HO or PARAM.ICIC.RL_state_pred_flag:
                    update_all_BS_L3_h_record(UE_list, large_fading, instant_channel, PARAM.L3_coe)

                '''更新UE的邻基站及其的L3测量'''
                find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading,
                                             instant_channel,
                                             PARAM.L3_coe)

                '''更新UE的服务基站L3测量'''
                update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)

                '''更新SS_SINR'''
                update_SS_SINR(UE_list, PARAM.sigma2, PARAM.filter_length_for_SINR)

            # for _UE in UE_list:
            #     if not _UE.active: continue
            #     if PARAM.ICIC.edge_divide_method == 'SINR':
            #         _UE.update_posi_type(PARAM.ICIC.SINR_th, PARAM.sigma2)
            #     else:
            #         _UE.posi_type = 'center'
            #         for edge_area_idx in range(PARAM.nCell - 1):
            #             if (edge_area_idx + 0.5) * PARAM.Dist - PARAM.ICIC.edge_area_width < np.real(_UE.posi) < (
            #                     edge_area_idx + 0.5) * PARAM.Dist + PARAM.ICIC.edge_area_width:
            #                 _UE.posi_type = 'edge'
            #                 break
            #     if _UE.serv_BS != -1 and _UE.posi_type != _UE.RB_type:
            #         _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
            #         _ = ICIC_RB_reallocate([_UE], _serv_BS, PARAM.RB_per_UE, serving_map)



        # '''若考虑边缘RB再利用，将额外的边缘RB资源重分配给一些中心用户'''
        # if PARAM.ICIC.edge_RB_reuse:
        #     for _UE in UE_list:
        #         if not _UE.active: continue
        #         if _UE.serv_BS != -1 and _UE.RB_type == 'center':
        #             _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
        #             _ = ICIC_edge_RB_reuse([_UE], _serv_BS, PARAM.RB_per_UE, serving_map)

        '''更新预编码信息和服务记录，10ms更新一次'''
        for _BS in BS_list:
            _BS.update_precoding_matrix(instant_channel, ZF_precoding)
            _BS.serv_UE_list_record.append(_BS.resource_map.serv_UE_list)
            _BS.RB_ocp_num_record.append(_BS.resource_map.RB_ocp_num)

        '''统计性能'''
        rec_P = get_receive_power(BS_list, instant_channel)
        inter_P = get_interference(BS_list, UE_list, instant_channel)
        SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma2)
        UE_rate = user_rate(PARAM.MLB.RB, SINR_dB, UE_list)
        # SNR_dB = calculate_SNR_dB(rec_P, PARAM.sigma2)

        max_inter_P = np.max(inter_P, axis=1)
        max_inter_list.append(max_inter_P)
        rate_list.append(UE_rate)

        '''每80ms做一次记录，保存数据'''
        if drop_idx % PARAM.posi_resolution == 0:
            if PARAM.ICIC.flag:
                center_UE, center_UE_offline, edge_UE, edge_UE_offline, UE_on_edge_RB = count_UE_offline(PARAM, UE_list, SINR_th=PARAM.ICIC.SINR_th_for_stat)
                center_UE_record.append(center_UE)
                center_UE_offline_record.append(center_UE_offline)
                edge_UE_record.append(edge_UE)
                edge_UE_offline_record.append(edge_UE_offline)
                UE_on_edge_RB_record.append(UE_on_edge_RB)
            else:
                active_UE, UE_offline = count_UE_offline(PARAM, UE_list, SINR_th=PARAM.ICIC.SINR_th_for_stat)
                UE_record.append(active_UE)
                UE_offline_record.append(UE_offline)

            if PARAM.ICIC.flag and PARAM.ICIC.dynamic:
                RB_for_edge_ratio_list.append(PARAM.ICIC.RB_for_edge_ratio)

        '''开始HO eval'''
        measure_criteria = 'L3'
        if PARAM.ICIC.flag == 1:
            allocate_method = ICIC_RB_allocate
        else:
            allocate_method = equal_RB_allocate

        if not PARAM.active_HO:
            # 被动HO
            handover_criteria_eval(PARAM, UE_list, BS_list, large_fading, instant_channel,
                                    serving_map, allocate_method, measure_criteria)
        else:
            actice_HO_eval(PARAM, NN, normalize_para, UE_list, BS_list, shadow, large_fading, instant_channel,
                                    serving_map, allocate_method, measure_criteria)

        '''对HO后SS_SINR变为None的UE估计SS_SINR'''
        update_SS_SINR(UE_list, PARAM.sigma2, PARAM.filter_length_for_SINR, after_HO=True)



        '''显示进度条'''
        progress_bar(drop_idx/(SimConfig.nDrop-1) * 100)

    HO_success = 0
    HO_failure = 0
    HO_failure_type_count = np.array([0,0,0,0])
    # failure_repeat_UE_num = 0
    for _UE in UE_list:
        HO_success = HO_success + _UE.HO_state.success_count
        HO_failure = HO_failure + _UE.HO_state.failure_count
        HO_failure_type_count += np.array(_UE.HO_state.failure_type_count)
        # if _UE.HO_state.failure_count > 1:
        #     failure_repeat_UE_num = failure_repeat_UE_num + 1
    print('HO results: HO success count: {}, HO failure count: {}, failure_type_count:{}'.format(HO_success, HO_failure, HO_failure_type_count))
    # return np.array(rate_list), np.array([HO_success, HO_failure, HO_failure_type_count])
    if PARAM.ICIC.flag:
        UE_offline_dict = {'center_UE':center_UE_record, 'center_UE_offline':center_UE_offline_record,
                             'edge_UE':edge_UE_record, 'edge_UE_offline':edge_UE_offline_record, 'UE_on_edge_RB':UE_on_edge_RB_record}
    else:
        UE_offline_dict = {'UE': UE_record, 'UE_offline': UE_offline_record}

    return np.array(rate_list), UE_list, BS_list, UE_offline_dict, np.array(max_inter_list), np.array(RB_for_edge_ratio_list)









if __name__ == '__main__':
    class SimConfig:  # 仿真参数
        save_flag = 1  # 是否保存结果
        root_path = 'result/0613_AHO_ICIC_pred_SINR'
        nDrop = 10000 - 10*8 # 时间步进长度

        # shadow_filepath = 'shadowFad_dB_8sigma_200dcov.mat'
        shadow_filepath = 'ShadowFad/0523_ShadowFad_dB_normed_6sigmaX_10dCov.mat'
        shadow_index = 'shadowFad_dB'

        # UE_posi_filepath = ['UE_tra/0514_scene0/Set_UE_posi_100s_500user_v{}.mat'.format(i+1) for i in range(3)]
        UE_posi_filepath = 'UE_tra/0527_scene0/Set_UE_posi_240.mat'
        posi_index = 'Set_UE_posi'

        '''大尺度信道预测模型'''
        model_name = 'scene0_large_h_DNN_0515'
        # model_name = 'DNN_0508'
        NN_path = 'Model/large_h_predict/'+model_name+'/'+model_name+'.dat'
        normalize_para_filename = 'Model/large_h_predict/'+model_name+'/normalize_para.npy'

    PARAM_list = []

    PARAM1 = Parameter()
    PARAM1.active_HO = True
    PARAM1.AHO.ideal_pred = False
    PARAM1.ICIC.flag = True
    PARAM1.ideal_RL_state = True
    PARAM1.RL_state_pred_flag = False
    PARAM1.RL_state_pred_len = 1  # max pred len refers to predictor
    PARAM1.dynamic_period = 1  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM1.nRB = 15
    PARAM_list.append(PARAM1)

    PARAM2 = Parameter()
    PARAM2.active_HO = True
    PARAM2.AHO.ideal_pred = False
    PARAM2.ICIC.flag = True
    PARAM2.ideal_RL_state = False
    PARAM2.RL_state_pred_flag = False
    PARAM2.RL_state_pred_len = 1  # max pred len refers to predictor
    PARAM2.dynamic_period = 1  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM2.nRB = 15
    PARAM_list.append(PARAM2)

    PARAM3 = Parameter()
    PARAM3.active_HO = True
    PARAM3.AHO.ideal_pred = False
    PARAM3.ICIC.flag = True
    PARAM3.ideal_RL_state = False
    PARAM3.RL_state_pred_flag = True
    PARAM3.RL_state_pred_len = 1  # max pred len refers to predictor
    PARAM3.dynamic_period = 1  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM3.nRB = 15
    PARAM_list.append(PARAM3)

    PARAM4 = Parameter()
    PARAM4.active_HO = True
    PARAM4.AHO.ideal_pred = False
    PARAM4.ICIC.flag = True
    PARAM4.ideal_RL_state = False
    PARAM4.RL_state_pred_flag = True
    PARAM4.RL_state_pred_len = 5  # max pred len refers to predictor
    PARAM4.dynamic_period = 5  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM4.nRB = 15
    PARAM_list.append(PARAM4)



    # PARAM5 = Parameter()
    # PARAM5.active_HO = True  # 主动切换
    # PARAM5.PHO.ideal_HO = False
    # PARAM5.AHO.ideal_pred = False
    # PARAM5.AHO.add_noise = False
    # PARAM5.ICIC.flag = True  # 干扰协调
    # PARAM5.ICIC.dynamic = True
    # PARAM5.nRB = 15
    # PARAM_list.append(PARAM5)

    # PARAM6 = Parameter()
    # PARAM6.active_HO = True  # 主动切换
    # PARAM6.PHO.ideal_HO = False
    # PARAM6.AHO.ideal_pred = False
    # PARAM6.AHO.add_noise = False
    # PARAM6.ICIC.flag = False
    # PARAM6.ICIC.dynamic = False
    # PARAM6.nRB = 50
    # PARAM_list.append(PARAM6)


    # for _SINR_th_for_stat in SINR_th_for_stat:
        # PARAM.HOM = _HOM
        # for _nRB in nRB_list:
        # _PARAM = copy.deepcopy(PARAM)
        # _PARAM.ICIC.SINR_th_for_stat = copy.deepcopy(_SINR_th_for_stat)

        # PARAM_list.append(_PARAM)
        # for _TTT in TTT_list:
        #     PARAM.TTT = _TTT
    # PARAM_list.append(copy.deepcopy(PARAM))

    def simulator_entry(PARAM_list, shadowFad_dB, UE_posi):
        if SimConfig.save_flag == 1:
            # _path = SimConfig.root_path+'/PARAM_list.npy'
            if not os.path.exists(SimConfig.root_path):
                os.makedirs(SimConfig.root_path)
            np.save(SimConfig.root_path+'/PARAM_list.npy', PARAM_list)

        _PARAM = PARAM_list[0]
        '''生成BS位置'''
        if _PARAM.scene == 0:
            Macro_Posi = road_cell_struct(_PARAM.nCell, _PARAM.Dist)
        else:
            Macro_Posi = cross_road_struction(_PARAM.Dist)

        '''开始仿真'''
        rate_list = []
        HO_result_list = []
        start_time = time.time()
        print('Simulation Start.\n')
        print('Important Parameters:')
        # print('Sigma: sigma2')
        print('Active HO: {}'.format(_PARAM.active_HO))
        if _PARAM.active_HO:
            print('Ideal Active HO: {}'.format(_PARAM.AHO.ideal_pred))
        print('ICIC: {}'.format(_PARAM.ICIC.flag))
        if _PARAM.ICIC.flag:
            print('Dynamic ICIC: {}'.format(_PARAM.ICIC.dynamic))
        print('Num of RB: {}\n'.format(_PARAM.nRB))

        for i in range(len(PARAM_list)):
            PARAM = PARAM_list[i]
            print('Simulation of Parameter Set:{} Start.'.format(i+1))
            if PARAM.active_HO and not PARAM.AHO.ideal_pred:
                obs_len = PARAM.AHO.obs_len
                pred_len = PARAM.AHO.pred_len
                nBS = PARAM.nCell
                NN = DNN_Model_Wrapper(input_dim=obs_len, output_dim=pred_len, no_units=100, learn_rate=0.001,
                                          batch_size=1000)
                NN.load(SimConfig.NN_path)

                normalize_para = np.load(SimConfig.normalize_para_filename, allow_pickle=True).tolist()
                NN.mean, NN.std = normalize_para['mean1'], normalize_para['sigma1']  # 读取归一化系数
            else:
                NN = None
                normalize_para = None

            Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map = init_all(PARAM, Macro_Posi, UE_posi, shadowFad_dB)
            _start_time = time.time()
            _rate_arr, _UE_list, _BS_list, _UE_offline_dict, _max_inter_arr, _RB_for_edge_ratio_arr = start_simulation(PARAM, Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map, NN, normalize_para)
            _end_time = time.time()
            print('Simulation of Parameter Set:{} Complete.'.format(i+1))
            print('Mean Rate:{:.2f} Mbps'.format(np.mean(_rate_arr[_rate_arr != 0])/1e6))
            print('Consumed Time:{:.2f}s\n'.format(_end_time - _start_time))
            rate_list.append(_rate_arr)
            if SimConfig.save_flag == 1:
                if not os.path.exists(SimConfig.root_path+'/{}'.format(i)):
                    os.makedirs(SimConfig.root_path+'/{}'.format(i))
                np.save(SimConfig.root_path + '/{}/rate_arr.npy'.format(i), _rate_arr)
                np.save(SimConfig.root_path + '/{}/UE_list.npy'.format(i), _UE_list)
                np.save(SimConfig.root_path + '/{}/BS_list.npy'.format(i), _BS_list)
                np.save(SimConfig.root_path + '/{}/UE_offline_dict.npy'.format(i), _UE_offline_dict)
                np.save(SimConfig.root_path + '/{}/max_inter_arr.npy'.format(i), _max_inter_arr)
                if PARAM.ICIC.flag and PARAM.ICIC.dynamic:
                    np.save(SimConfig.root_path + '/{}/RB_for_edge_ratio_arr'.format(i), _RB_for_edge_ratio_arr)
            # HO_result_list.append(_HO_result)

        end_time = time.time()
        print('All Simulation Complete.')
        print('Total Consumed Time:{:.2f}s\n'.format(end_time - start_time))

        return


    np.random.seed(0)
    '''从文件读取阴影衰落'''
    shadowFad_dB = get_shadow_from_mat(SimConfig.shadow_filepath, SimConfig.shadow_index)

    '''从文件读取UE位置'''
    # filepath = 'Set_UE_posi_60s_250user_1to2_new1.mat'
    UE_posi = get_UE_posi_from_file(SimConfig.UE_posi_filepath, SimConfig.posi_index)
    # UE_posi = UE_posi[2, :, :]

    if len(UE_posi.shape) != 3:
        UE_posi = np.swapaxes(UE_posi, 0,1)
        UE_posi = np.reshape(UE_posi, (PARAM_list[0].ntype, -1, UE_posi.shape[1]))
        UE_posi = np.swapaxes(UE_posi, 1, 2)

    UE_posi = process_posi_data(UE_posi)

    '''进入仿真'''
    simulator_entry(PARAM_list, shadowFad_dB, UE_posi)

