'''
本模块为仿真器入口
'''


import tensorflow as tf
import pickle
from ReinfocementLearning.actor import ActorNetwork
import tf_slim as slim
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
                     instant_channel:InstantChannelMap, serving_map:ServingMap, NN=None, normalize_para=None, actor=None, sess=None, extra_interf_map=None):
    '''开始仿真'''

    '''更新UE的邻基站'''
    find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading, instant_channel, PARAM.L3_coe)

    '''统计小区内UE'''
    count_UE_in_range(UE_list, BS_list)

    '''初始RL state'''
    update_SS_SINR(UE_list, BS_list, PARAM.sigma2, PARAM.filter_length_for_SINR)


    '''若考虑干扰协调，划分边缘用户'''
    if PARAM.ICIC.flag:
        if PARAM.ICIC.dynamic:
            dynamic_nRB_per_UE_and_ICIC(PARAM, BS_list, UE_list, actor, sess)
        ICIC_decide_edge_UE(PARAM, BS_list, UE_list, init_flag=True)

    '''初始接入'''
    _ = access_init(PARAM, BS_list, UE_list, instant_channel, serving_map)

    '''更新所有基站的L3测量（预测大尺度信道时需要）'''
    if PARAM.active_HO or PARAM.ICIC.RL_state_pred_flag:
        update_all_BS_L3_h_record(UE_list, large_fading, instant_channel, PARAM.L3_coe)



    '''若考虑RB干扰协调，根据UE类型重新分配RB'''
    if PARAM.ICIC.flag:
        for _BS in BS_list:
            ICIC_BS_RB_allocate(UE_list, _BS, serving_map)


    '''更新UE的服务基站L3测量'''
    update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)

    '''更新预编码信息和服务记录'''
    for _BS in BS_list:
        _BS.update_precoding_matrix(instant_channel, ZF_precoding)
        _BS.serv_UE_list_record.append(_BS.resource_map.serv_UE_list)
        _BS.RB_ocp_num_record.append(_BS.resource_map.RB_ocp_num)

    '''接入时SINR和速率'''
    rec_P = get_receive_power(BS_list, instant_channel)
    inter_P = get_interference(PARAM, BS_list, UE_list, instant_channel, extra_interf_map)
    SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma2)
    UE_rate = user_rate(PARAM.MLB.RB, SINR_dB, UE_list)
    # print(np.mean(UE_rate))
    max_inter_P = np.max(inter_P, axis=1)
    max_inter_list = [max_inter_P]
    rate_list = [UE_rate]

    _center_cell_rate = np.zeros((len(UE_list),))
    for _UE in UE_list:
        if _UE.serv_BS == 2:
            _center_cell_rate[_UE.no] = UE_rate[_UE.no]
    center_cell_rate_list = [_center_cell_rate]

    '''不满足GRB的用户数'''
    unsatisfied_GBR_num = []

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

        '''以下操作均以80ms为步长'''
        if drop_idx % PARAM.time_resolution == 0:
            '''更新UE位置'''
            _posi_idx = int(drop_idx // PARAM.time_resolution)
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
            if drop_idx % PARAM.time_resolution == 0:
                large_h = large_scale_channel(PARAM, BS_list, UE_list, shadow)
                large_fading.update(large_h)

            '''更新瞬时信道信息'''
            instant_channel.calculate_by_fading(large_fading, small_fading)

            '''更新UE的邻基站'''
            find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading, instant_channel,
                                         PARAM.L3_coe)

            '''统计小区内UE'''
            count_UE_in_range(UE_list, BS_list)

            '''更新RL state'''
            update_SS_SINR(UE_list, BS_list, PARAM.sigma2, PARAM.filter_length_for_SINR)

            '''活动UE尝试进行接入，停止活动的UE断开,并更新RL state'''
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
                update_serv_BS_L3_h([_UE], large_fading, instant_channel, PARAM.L3_coe)
                update_SS_SINR([_UE], BS_list, PARAM.sigma2, PARAM.filter_length_for_SINR)


            '''更新所有基站的L3测量（预测大尺度信道时需要）'''
            if PARAM.active_HO:
                update_all_BS_L3_h_record(UE_list, large_fading, instant_channel, PARAM.L3_coe)

            '''更新UE的邻基站及其的L3测量'''
            find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading,
                                             instant_channel, PARAM.L3_coe)

            '''更新UE的服务基站L3测量'''
            update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)

            '''更新SS_SINR'''
            update_SS_SINR(UE_list, BS_list, PARAM.sigma2, PARAM.filter_length_for_SINR)


            if PARAM.ICIC.RL_state_pred_flag:
                update_pred_SS_SINR(UE_list, PARAM.sigma2, NN, normalize_para, PARAM.ICIC.RL_state_pred_len)



            '''若考虑干扰协调，划分边缘用户'''
            if PARAM.ICIC.flag:
                if PARAM.ICIC.dynamic and (drop_idx / PARAM.time_resolution) % PARAM.ICIC.dynamic_period == 0:
                    dynamic_nRB_per_UE_and_ICIC(PARAM, BS_list, UE_list, actor, sess)
                ICIC_decide_edge_UE(PARAM, BS_list, UE_list)



            '''若考虑RB干扰协调，根据UE类型分配RB'''
            if PARAM.ICIC.flag:
                # '''针对使用不合法RB的用户和RB类型不一致的用户，重新分配RB'''
                # ICIC_RB_reallocate(UE_list, BS_list, serving_map)

                '''对BS内UE做RB分配'''
                for _BS in BS_list:
                    ICIC_BS_RB_allocate(UE_list, _BS, serving_map)  # 边缘UE优先分配正交资源

            '''更新UE的服务基站L3测量'''
            update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)



        '''更新预编码信息和服务记录，10ms更新一次'''
        for _BS in BS_list:
            _BS.update_precoding_matrix(instant_channel, ZF_precoding)
            _BS.serv_UE_list_record.append(_BS.resource_map.serv_UE_list)
            _BS.RB_ocp_num_record.append(_BS.resource_map.RB_ocp_num)

        '''统计性能'''
        rec_P = get_receive_power(BS_list, instant_channel)
        inter_P = get_interference(PARAM, BS_list, UE_list, instant_channel, extra_interf_map)
        SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma2)
        UE_rate = user_rate(PARAM.MLB.RB, SINR_dB, UE_list)
        # SNR_dB = calculate_SNR_dB(rec_P, PARAM.sigma2)

        max_inter_P = np.max(inter_P, axis=1)
        max_inter_list.append(max_inter_P)
        rate_list.append(UE_rate)



        '''每80ms做一次记录，保存数据'''
        _unstf_GBR_num = 0
        _center_cell_rate = np.zeros((len(UE_list),))
        if drop_idx % PARAM.time_resolution == 0:
            '''统计没有达到最低速率的GBR用户'''
            for _UE in UE_list:
                if _UE.GBR_flag and _UE.active:
                    _rate = UE_rate[_UE.no]
                    if _rate < _UE.min_rate:
                        _unstf_GBR_num = _unstf_GBR_num+1
                if _UE.serv_BS == 2:
                    _center_cell_rate[_UE.no] = UE_rate[_UE.no]
            center_cell_rate_list.append(_center_cell_rate)
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
        count_UE_in_range(UE_list, BS_list)
        update_SS_SINR(UE_list, BS_list, PARAM.sigma2, PARAM.filter_length_for_SINR, after_HO=True)



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

    if PARAM.GBR_ratio != 0:
        print('Unsatified GBR UE count: {}'.format(np.sum(unsatisfied_GBR_num)))

    return np.array(rate_list), UE_list, BS_list, UE_offline_dict, np.array(max_inter_list), np.array(RB_for_edge_ratio_list), np.array(center_cell_rate_list)









if __name__ == '__main__':
    class SimConfig:  # 仿真参数
        save_flag = 0  # 是否保存结果
        root_path = 'result/0708_7BS_add_extra_itf'
        nDrop = 6000 - 10*8  # 时间步进长度

        # shadow_filepath = 'shadowFad_dB_8sigma_200dcov.mat'
        # shadow_filepath = 'ShadowFad/0523_ShadowFad_dB_normed_6sigmaX_10dCov.mat'
        shadow_filepath = 'ShadowFad/0627_7BS_ShadowFad_dB_normed_6sigmaX_10dCov.mat'
        shadow_index = 'shadowFad_dB'

        # UE_posi_filepath = ['UE_tra/0514_scene0/Set_UE_posi_100s_500user_v{}.mat'.format(i+1) for i in range(3)]
        # UE_posi_filepath = 'UE_tra/0621_2row_15BS/Set_UE_posi_150user_15BS.mat'
        # UE_posi_filepath = 'UE_tra/0527_scene0/Set_UE_posi_180.mat'
        UE_posi_filepath = 'UE_tra/0627_7road_7BS/Set_UE_posi_60s_330user_7BS_V123.mat'
        posi_index = 'Set_UE_posi'

        '''额外干扰地图'''
        extra_itf_filepath = '7BS_outer_itf_map.mat'
        extra_itf_index = 'itf_map'

        '''大尺度信道预测模型'''
        model_name = 'scene0_large_h_DNN_0515'
        # model_name = 'DNN_0508'
        NN_path = 'Model/large_h_predict/'+model_name+'/'+model_name+'.dat'
        normalize_para_filename = 'Model/large_h_predict/'+model_name+'/normalize_para.npy'

        '''每个UE的RB数决策模型'''
        actor_path = 'ReinfocementLearning/models/0706model/actor_1234567_G3_0705_01'
        actor_lr = 0.001
        actor_num_hidden_1 = 50
        actor_num_hidden_2 = 40
        actor_num_hidden_3 = 30
        actor_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        actor_config.gpu_options.allow_growth = True
        # actor = ActorNetwork(actor_lr, actor_num_hidden_1, actor_num_hidden_2, actor_num_hidden_3)


    PARAM_list = []

    # PARAM0 = Parameter()
    # PARAM0.active_HO = True  # 主动切换
    # PARAM0.AHO.ideal_pred = False
    # # PARAM0.ICIC.flag = True
    # PARAM0.ICIC.RB_partition_num = 3
    # PARAM0.ICIC.dynamic = True  # ICIC
    # PARAM0.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM0.ICIC.obsolete_time = 10
    # PARAM0.ICIC.RL_state_pred_flag = False
    # # PARAM0.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM0.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM0.dynamic_nRB_per_UE = True  # 动态调整
    # PARAM0.RB_per_UE = 3
    # PARAM0.nRB = 30
    # PARAM_list.append(PARAM0)

    # PARAM1 = Parameter()
    # PARAM1.active_HO = False  # 被动切换
    # PARAM1.AHO.ideal_pred = False
    # # PARAM1.ICIC.flag = True
    # PARAM1.ICIC.dynamic = False
    # PARAM1.ICIC.RB_for_edge_ratio = 0  # 不做ICIC
    # PARAM1.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM1.ICIC.obsolete_time = 10
    # PARAM1.ICIC.RL_state_pred_flag = False
    # # PARAM1.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM1.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM1.dynamic_nRB_per_UE = False
    # PARAM1.RB_per_UE = 3
    # PARAM1.nRB = 30
    # PARAM_list.append(PARAM1)


    PARAM2 = Parameter()
    PARAM2.active_HO = True  # 主动切换
    PARAM2.AHO.ideal_pred = False
    # PARAM2.ICIC.flag = True
    PARAM2.ICIC.dynamic = True  # ICIC
    PARAM2.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # PARAM2.ICIC.obsolete_time = 10
    PARAM2.ICIC.RL_state_pred_flag = False
    # PARAM2.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # PARAM2.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM2.dynamic_nRB_per_UE = False
    PARAM2.RB_per_UE = 3
    PARAM2.nRB = 30
    PARAM_list.append(PARAM2)
    #
    #
    PARAM3 = Parameter()
    PARAM3.active_HO = False  # 被动切换
    PARAM3.AHO.ideal_pred = False
    # PARAM3.ICIC.flag = True
    PARAM3.ICIC.dynamic = False
    PARAM3.ICIC.RB_for_edge_ratio = 0  # 不做ICIC
    PARAM3.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # PARAM3.ICIC.obsolete_time = 10
    PARAM3.ICIC.RL_state_pred_flag = False
    # PARAM3.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # PARAM3.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM3.dynamic_nRB_per_UE = False
    PARAM3.RB_per_UE = 6
    # PARAM3.nRB = 15
    PARAM_list.append(PARAM3)


    PARAM4 = Parameter()
    PARAM4.active_HO = True  # 主动切换
    PARAM4.AHO.ideal_pred = False
    # PARAM4.ICIC.flag = True
    PARAM4.ICIC.dynamic = True  # ICIC
    PARAM4.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # PARAM4.ICIC.obsolete_time = 10
    PARAM4.ICIC.RL_state_pred_flag = False
    # PARAM4.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # PARAM4.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM4.dynamic_nRB_per_UE = False
    PARAM4.RB_per_UE = 6
    # PARAM4.nRB = 15
    PARAM_list.append(PARAM4)


    PARAM5 = Parameter()
    PARAM5.active_HO = False  # 被动切换
    PARAM5.AHO.ideal_pred = False
    # PARAM5.ICIC.flag = True
    PARAM5.ICIC.dynamic = False
    PARAM5.ICIC.RB_for_edge_ratio = 0  # 不做ICIC
    PARAM5.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # PARAM5.ICIC.obsolete_time = 10  # 过时10帧
    PARAM5.ICIC.RL_state_pred_flag = False
    # PARAM5.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # PARAM5.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM5.dynamic_nRB_per_UE = False
    PARAM5.RB_per_UE = 9
    # PARAM5.nRB = 15
    PARAM_list.append(PARAM5)


    PARAM6 = Parameter()
    PARAM6.active_HO = True  # 主动切换
    PARAM6.AHO.ideal_pred = False
    # PARAM6.ICIC.flag = True
    PARAM6.ICIC.dynamic = True  # ICIC
    PARAM6.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # PARAM6.ICIC.obsolete_time = 10  # 过时10帧
    PARAM6.ICIC.RL_state_pred_flag = False
    # PARAM6.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # PARAM6.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM6.dynamic_nRB_per_UE = False
    PARAM6.RB_per_UE = 9
    # PARAM6.nRB = 15
    PARAM_list.append(PARAM6)

    # PARAM7 = Parameter()
    # PARAM7.active_HO = False  # 被动切换
    # PARAM7.AHO.ideal_pred = False
    # # PARAM7.ICIC.flag = True
    # PARAM7.ICIC.dynamic = False
    # PARAM7.ICIC.RB_for_edge_ratio = 0  # 不做ICIC
    # PARAM7.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM7.ICIC.obsolete_time = 10  # 过时10帧
    # PARAM7.ICIC.RL_state_pred_flag = False
    # # PARAM7.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM7.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM7.dynamic_nRB_per_UE = False
    # PARAM7.RB_per_UE = 12
    # # PARAM7.nRB = 15
    # PARAM_list.append(PARAM7)
    #
    #
    # PARAM8 = Parameter()
    # PARAM8.active_HO = True  # 主动切换
    # PARAM8.AHO.ideal_pred = False
    # # PARAM8.ICIC.flag = True
    # PARAM8.ICIC.dynamic = True  # ICIC
    # PARAM8.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM8.ICIC.obsolete_time = 10  # 过时10帧
    # PARAM8.ICIC.RL_state_pred_flag = False
    # # PARAM8.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM8.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM8.dynamic_nRB_per_UE = False
    # PARAM8.RB_per_UE = 12
    # # PARAM8.nRB = 15
    # PARAM_list.append(PARAM8)


    # for _SINR_th_for_stat in SINR_th_for_stat:
        # PARAM.HOM = _HOM
        # for _nRB in nRB_list:
        # _PARAM = copy.deepcopy(PARAM)
        # _PARAM.ICIC.SINR_th_for_stat = copy.deepcopy(_SINR_th_for_stat)

        # PARAM_list.append(_PARAM)
        # for _TTT in TTT_list:
        #     PARAM.TTT = _TTT
    # PARAM_list.append(copy.deepcopy(PARAM))

    def simulator_entry(PARAM_list, shadowFad_dB, UE_posi, extra_interf_map=None):
        if SimConfig.save_flag == 1:
            # _path = SimConfig.root_path+'/PARAM_list.npy'
            if not os.path.exists(SimConfig.root_path):
                os.makedirs(SimConfig.root_path)
            np.save(SimConfig.root_path+'/PARAM_list.npy', PARAM_list)

        _PARAM = PARAM_list[0]
        '''生成BS位置'''
        if _PARAM.scene == 0:
            Macro_Posi = road_cell_struct(_PARAM.nCell, _PARAM.Dist)
        elif _PARAM.scene == 1:
            Macro_Posi = cross_road_struction(_PARAM.Dist)
        else:
            Macro_Posi = cell_struction(_PARAM.nCell, _PARAM.Dist)

        # plot_BS_location(Macro_Posi)
        # plt.show()

        '''开始仿真'''
        rate_list = []
        HO_result_list = []
        start_time = time.time()
        print('Simulation Start.\n')


        for i in range(len(PARAM_list)):
            PARAM = PARAM_list[i]
            print('Simulation of Parameter Set:{} Start.'.format(i+1))
            print('Important Parameters:')
            # print('Sigma: sigma2')
            print('Active HO: {}'.format(PARAM.active_HO))
            if PARAM.active_HO:
                print('Ideal Active HO: {}'.format(PARAM.AHO.ideal_pred))
            print('ICIC: {}'.format(PARAM.ICIC.flag))
            if PARAM.ICIC.flag:
                print('Dynamic ICIC: {}'.format(PARAM.ICIC.dynamic))
            print('Num of RB: {}\n'.format(PARAM.nRB))
            if PARAM.dynamic_nRB_per_UE:
                print('Dynamic nRB for UE')

            '''大尺度信道预测模型'''
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

            '''每个UE的RB数决策模型'''
            if PARAM.dynamic_nRB_per_UE:
                def load_model(fname):
                    f = open(fname, 'rb')
                    var = pickle.load(f)
                    f.close()
                    return var
                actor = ActorNetwork(SimConfig.actor_lr, SimConfig.actor_num_hidden_1, SimConfig.actor_num_hidden_2, SimConfig.actor_num_hidden_3)
                sess = tf.compat.v1.Session(config=SimConfig.actor_config)
                sess.run(tf.compat.v1.global_variables_initializer())
                global_vars = load_model(SimConfig.actor_path)
                actor.set_actor_variable(sess, global_vars)
            else:
                actor = None
                sess = None

            Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map = init_all(PARAM, Macro_Posi, UE_posi, shadowFad_dB)

            '''进入仿真'''
            _start_time = time.time()
            _rate_arr, _UE_list, _BS_list, _UE_offline_dict, _max_inter_arr, _RB_for_edge_ratio_arr, _center_cell_rate_arr = start_simulation(PARAM, Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map, NN, normalize_para, actor, sess, extra_interf_map)
            _end_time = time.time()

            '''一轮仿真结束，输出和保存信息'''
            print('Simulation of Parameter Set:{} Complete.'.format(i+1))
            print('Mean Rate:{:.2f} Mbps'.format(np.mean(_rate_arr[_rate_arr != 0])/1e6))
            print('Center Cell Mean Rate:{:.2f} Mbps'.format(np.mean(_center_cell_rate_arr[_center_cell_rate_arr != 0]) / 1e6))
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
                np.save(SimConfig.root_path + '/{}/center_cell_rate_arr.npy'.format(i), _center_cell_rate_arr)
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
    UE_posi = get_UE_posi_from_file(SimConfig.UE_posi_filepath, SimConfig.posi_index)
    # UE_posi = UE_posi[2, :, :]

    '''调整UE轨迹的数据为【行人，自行车，汽车】'''
    if len(UE_posi.shape) != 3:
        UE_posi = np.swapaxes(UE_posi, 0,1)
        # up_v0_UE_posi = UE_posi[:30,:]
        # up_v1_UE_posi = UE_posi[30:60,:]
        # up_v2_UE_posi = UE_posi[60:90,:]
        # down_v0_UE_posi = UE_posi[90:120,:]
        # down_v1_UE_posi = UE_posi[120:150,:]
        # down_v2_UE_posi = UE_posi[150:180,:]
        # UE_posi = np.concatenate((up_v0_UE_posi,down_v0_UE_posi,up_v1_UE_posi, down_v1_UE_posi,up_v2_UE_posi,down_v2_UE_posi), axis=0)
        UE_posi = np.reshape(UE_posi, (PARAM_list[0].ntype, -1, UE_posi.shape[1]))
        UE_posi = np.swapaxes(UE_posi, 1, 2)

    UE_posi = process_posi_data(UE_posi)

    # '''验证UE轨迹'''
    # fig, ax = plt.subplots()
    # for i in range(8):
    #     _UE_tra = UE_posi[0][:,i*5]
    #     real_part = np.real(_UE_tra.tolist())
    #     imag_part = np.imag(_UE_tra.tolist())
    #     ax.plot(real_part, imag_part)
    # plt.show()

    '''从文件读取额外干扰地图'''
    try:
        extra_interf_map = get_UE_posi_from_file(SimConfig.extra_itf_filepath, SimConfig.extra_itf_index)
    except:
        extra_interf_map = None

    '''进入仿真'''
    simulator_entry(PARAM_list, shadowFad_dB, UE_posi, extra_interf_map)

