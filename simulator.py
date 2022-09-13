'''
本模块为仿真器入口
'''


import tensorflow as tf
import pickle
# from ReinforcementLearningV1.actor import ActorNetwork
from DNN_model_utils import DNN_Model_Wrapper
from unsupervised0829.actor import ActorNetwork
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
    update_SS_SINR(PARAM, UE_list, BS_list, extra_interf_map=extra_interf_map)

    '''若考虑干扰协调，划分边缘用户'''
    if PARAM.ICIC.flag:
        if PARAM.ICIC.dynamic:
            dynamic_nRB_per_UE_and_ICIC(PARAM, BS_list, UE_list, serving_map, large_fading, instant_channel,
                                        extra_interf_map, PARAM.dynamic_nRB_scheme, actor, sess)
        ICIC_decide_edge_UE(PARAM, BS_list, UE_list, init_flag=True)


    '''初始接入'''
    _ = access_init(PARAM, BS_list, UE_list, instant_channel, serving_map)

    '''更新RL state'''
    update_SS_SINR(PARAM, UE_list, BS_list, extra_interf_map=extra_interf_map)


    '''更新所有基站的L3测量（预测大尺度信道时需要）'''
    if PARAM.active_HO or PARAM.ICIC.RL_state_pred_flag:
        update_all_BS_L3_h_record(UE_list, large_fading, instant_channel, PARAM.L3_coe)



    '''若考虑RB干扰协调，根据UE类型重新分配RB'''
    if PARAM.ICIC.flag:
        for _BS in BS_list:
            ICIC_BS_RB_allocate(PARAM, UE_list, _BS, serving_map)


    '''更新UE的服务基站L3测量'''
    update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)

    '''更新预编码信息和服务记录'''
    for _BS in BS_list:
        _BS.update_precoding_matrix(instant_channel, MMSE_precoding)
        _BS.serv_UE_list_record.append(_BS.resource_map.serv_UE_list)
        _BS.RB_ocp_num_record.append(_BS.resource_map.RB_ocp_num)

    '''接入时SINR和速率'''
    rec_P = get_receive_power(BS_list, instant_channel)
    inter_P = get_interference(PARAM, BS_list, UE_list, instant_channel, extra_interf_map)
    SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma2)
    UE_rate = user_rate(PARAM.MLB.RB, SINR_dB, UE_list)
    # print(np.mean(UE_rate))

    rec_P_list = [rec_P]
    inter_P_list = [inter_P]
    # max_inter_P = np.max(inter_P, axis=1)
    # max_inter_list = [max_inter_P]
    rate_list = [UE_rate]

    est_rec_power = []
    est_itf_power = []
    est_ICIC_itf_power = []
    _UE_in_different_cell = [[] for _ in range(len(BS_list))]
    for _UE in UE_list:
        if _UE.active:
            _UE_in_different_cell[_UE.serv_BS].append(_UE.no)
            est_rec_power.append(_UE.RL_state.estimated_rec_power)
            est_itf_power.append(_UE.RL_state.estimated_itf_power)
            est_ICIC_itf_power.append(_UE.RL_state.estimated_ICIC_itf_power)

    UE_in_different_cell_list = [_UE_in_different_cell]
    est_rec_power_list = [est_rec_power]
    est_itf_power_list = [est_itf_power]
    est_ICIC_itf_power_list = [est_ICIC_itf_power]

    '''GBR用户'''
    _GBR_UE = []
    for _UE in UE_list:
        if _UE.GBR_flag and _UE.active:
            _GBR_UE.append(_UE.no)
    GBR_UE_list = [_GBR_UE]

    '''统计中心、边缘UE数、未服务率'''
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
        if drop_idx >= 8:
            probe = 8
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
                    _future_posi = UE_posi[_posi_idx + 1:_posi_idx + 1 + _UE.record_len, _UE.type_no]
                    _UE.update_future_posi(_future_posi)
                elif len(UE_posi.shape) == 3:
                    _UE_posi = UE_posi[_UE.type, _posi_idx, _UE.type_no]
                    _UE.update_posi(_UE_posi)
                    _future_posi = UE_posi[_UE.type][_posi_idx + 1:_posi_idx + 1 + _UE.record_len, _UE.type_no]
                    _UE.update_future_posi(_future_posi)

            '''更新小尺度信道信息'''
            small_h = small_scale_fading(len(BS_list), PARAM.nUE, PARAM.nRB, PARAM.Macro.nNt)
            small_fading.update(small_h)

            '''更新大尺度信道信息'''
            large_h = large_scale_channel(PARAM, BS_list, UE_list, shadow)
            large_fading.update(large_h)

            '''更新瞬时信道信息'''
            instant_channel.calculate_by_fading(large_fading, small_fading)

            '''更新UE的邻基站'''
            find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading, instant_channel,
                                             PARAM.L3_coe)

        '''切换'''
        '''更新小尺度信道信息'''
        small_h = small_scale_fading(len(BS_list), PARAM.nUE, PARAM.nRB, PARAM.Macro.nNt)
        small_fading.update(small_h)

        # '''更新大尺度信道信息'''
        # large_h = large_scale_channel(PARAM, BS_list, UE_list, shadow)
        # large_fading.update(large_h)

        '''更新瞬时信道信息'''
        instant_channel.calculate_by_fading(large_fading, small_fading)

        if PARAM.PHO.ideal_HO:
            if drop_idx % PARAM.time_resolution == 0:
                '''初始接入'''
                _ = access_init(PARAM, BS_list, UE_list, instant_channel, serving_map)

                '''更新RL state'''
                update_SS_SINR(PARAM, UE_list, BS_list, extra_interf_map=extra_interf_map)

                '''更新所有基站的L3测量（预测大尺度信道时需要）'''
                if PARAM.active_HO:
                    update_all_BS_L3_h_record(UE_list, large_fading, instant_channel, PARAM.L3_coe)

                '''更新UE的邻基站及其的L3测量'''
                find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading,
                                             instant_channel, PARAM.L3_coe)

                '''更新UE的服务基站L3测量'''
                update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)

                '''更新SS_SINR'''
                update_SS_SINR(PARAM, UE_list, BS_list, extra_interf_map=extra_interf_map)
        else:
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
                # 主动HO
                actice_HO_eval(PARAM, NN, normalize_para, UE_list, BS_list, shadow, large_fading, instant_channel,
                                        serving_map, allocate_method, measure_criteria)

            '''对HO后SS_SINR变为None的UE估计SS_SINR'''
            count_UE_in_range(UE_list, BS_list)
            update_SS_SINR(PARAM, UE_list, BS_list, after_HO=True, extra_interf_map=extra_interf_map)


        '''以下操作均以80ms为步长'''
        if drop_idx % PARAM.time_resolution == 0:
            '''统计小区内UE'''
            count_UE_in_range(UE_list, BS_list)

            '''更新RL state'''
            update_SS_SINR(PARAM, UE_list, BS_list, extra_interf_map=extra_interf_map)


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
                update_SS_SINR(PARAM, [_UE], BS_list, extra_interf_map=extra_interf_map)


            '''更新所有基站的L3测量（预测大尺度信道时需要）'''
            if PARAM.active_HO:
                update_all_BS_L3_h_record(UE_list, large_fading, instant_channel, PARAM.L3_coe)

            '''更新UE的邻基站及其的L3测量'''
            find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading,
                                             instant_channel, PARAM.L3_coe)

            '''更新UE的服务基站L3测量'''
            update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)

            '''更新SS_SINR'''
            update_SS_SINR(PARAM, UE_list, BS_list, extra_interf_map=extra_interf_map)


            if PARAM.ICIC.RL_state_pred_flag:
                update_pred_SS_SINR(UE_list, PARAM.sigma2, NN, normalize_para, PARAM.ICIC.RL_state_pred_len)


            '''若考虑干扰协调，确定策略'''
            if PARAM.ICIC.flag:
                if PARAM.ICIC.dynamic and (drop_idx / PARAM.time_resolution) % PARAM.ICIC.dynamic_period == 0:
                    dynamic_nRB_per_UE_and_ICIC(PARAM, BS_list, UE_list, serving_map, large_fading, instant_channel,
                                                extra_interf_map, PARAM.dynamic_nRB_scheme, actor, sess)
                ICIC_decide_edge_UE(PARAM, BS_list, UE_list)


            '''若考虑RB干扰协调，根据UE类型分配RB'''
            if PARAM.ICIC.flag:
                # '''针对使用不合法RB的用户和RB类型不一致的用户，重新分配RB'''
                # ICIC_RB_reallocate(UE_list, BS_list, serving_map)
                '''对BS内UE做RB分配'''
                for _BS in BS_list:
                    ICIC_BS_RB_allocate(PARAM, UE_list, _BS, serving_map)  # 边缘UE优先分配正交资源

            '''更新UE的服务基站L3测量'''
            update_serv_BS_L3_h(UE_list, large_fading, instant_channel, PARAM.L3_coe)


        '''以下10ms更新一次'''

        '''更新预编码信息和服务记录'''
        for _BS in BS_list:
            _BS.update_precoding_matrix(instant_channel, MMSE_precoding)
            _BS.serv_UE_list_record.append(_BS.resource_map.serv_UE_list)
            _BS.RB_ocp_num_record.append(_BS.resource_map.RB_ocp_num)

        '''统计性能'''
        rec_P = get_receive_power(BS_list, instant_channel)
        inter_P = get_interference(PARAM, BS_list, UE_list, instant_channel, extra_interf_map)
        SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma2)
        UE_rate = user_rate(PARAM.MLB.RB, SINR_dB, UE_list)
        # SNR_dB = calculate_SNR_dB(rec_P, PARAM.sigma2)
        rec_P_list.append(rec_P)
        inter_P_list.append(inter_P)
        # max_inter_P = np.max(inter_P, axis=1)
        # max_inter_list.append(max_inter_P)
        rate_list.append(UE_rate)

        if len(rate_list) == 8*50:
            probe = 50




        '''每80ms做一次记录，保存数据'''
        if drop_idx % PARAM.time_resolution == 0:
            '''统计'''
            _GBR_UE = []
            _UE_in_different_cell = [[] for _ in range(len(BS_list))]
            est_rec_power = []
            est_itf_power = []
            est_ICIC_itf_power = []
            for _UE in UE_list:
                if _UE.GBR_flag and _UE.active:
                    _GBR_UE.append(_UE.no)
                if _UE.active:
                    # _center_cell_rate[_UE.no] = UE_rate[_UE.no]
                    # _center_cell_UE.append(_UE.no)
                    _UE_in_different_cell[_UE.serv_BS].append(_UE.no)
                    est_rec_power.append(_UE.RL_state.estimated_rec_power)
                    est_itf_power.append(_UE.RL_state.estimated_itf_power)
                    est_ICIC_itf_power.append(_UE.RL_state.estimated_ICIC_itf_power)

            UE_in_different_cell_list.append(_UE_in_different_cell)
            est_rec_power_list.append(est_rec_power)
            est_itf_power_list.append(est_itf_power)
            est_ICIC_itf_power_list.append(est_ICIC_itf_power)

            GBR_UE_list.append(_GBR_UE)
            # center_cell_UE_list.append(_center_cell_UE)


            '''统计中心、边缘用户以及未服务用户'''
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

    est_RL_state_dict = {'est_rec_power':est_rec_power_list, 'est_itf_power':est_itf_power_list, 'est_ICIC_itf_power':est_ICIC_itf_power_list}

    return np.array(rate_list), UE_list, BS_list, UE_offline_dict, np.array(rec_P_list), np.array(inter_P_list), \
           np.array(RB_for_edge_ratio_list), np.array(UE_in_different_cell_list), np.array(GBR_UE_list), est_RL_state_dict









if __name__ == '__main__':
    class SimConfig:  # 仿真参数
        save_flag = 1  # 是否保存结果
        root_path = 'result/0910_0.10GBR2M(+3RB)_02Set_UE_posi_330user_edge_7BS_liji_duobu_bylarge+fixednRB'
        # nDrop = 6000 - 10*8  # 时间步进长度
        nDrop = 400
        # shadow_filepath = 'shadowFad_dB_8sigma_200dcov.mat'
        # shadow_filepath = 'ShadowFad/0523_ShadowFad_dB_normed_6sigmaX_10dCov.mat'
        shadow_filepath = 'ShadowFad/0627_7BS_ShadowFad_dB_normed_6sigmaX_10dCov.mat'
        shadow_index = 'shadowFad_dB'

        # UE_posi_filepath = ['UE_tra/0514_scene0/Set_UE_posi_100s_500user_v{}.mat'.format(i+1) for i in range(3)]
        # UE_posi_filepath = 'UE_tra/0621_2row_15BS/Set_UE_posi_150user_15BS.mat'
        # UE_posi_filepath = 'UE_tra/0527_scene0/Set_UE_posi_180.mat'
        # UE_posi_filepath = 'UE_tra/0714_7road_7BS/Set_UE_posi_60s_330user_7BS_V123.mat'
        # UE_posi_filepath = 'UE_tra/0726_7BS_withstaticUE/Set_UE_posi_60s_330user_7BS_150move_180static.mat'
        # UE_posi_filepath = 'UE_tra/0726_7BS_withstaticUE/0727edge2.mat'
        # UE_posi_filepath = 'UE_tra/0721_7BS_PPP/Set_UE_posi_PPP_60s_[10,90,30,10,90,90,10]user_7BS_.mat'
        # UE_posi_filepath = 'UE_tra/0721_7BS_PPP/Set_UE_posi_PPP_60s_60-75m_[80,20,50,70,30,30,50]user_7BS_.mat'
        # UE_posi_filepath = 'UE_tra/0721_7BS_PPP/Set_UE_posi_50_330user_balanced_edge_7BS.mat'
        # UE_posi_filepath = 'UE_tra/0726_7BS_withstaticUE/Set_UE_posi_60s_180moving_150static_7BS_.mat'
        # UE_posi_filepath = 'UE_tra/0714_7road_7BS/Set_UE_posi_60s_432user_7BS_V123.mat'
        # UE_posi_filepath = 'UE_tra/0721_7BS_PPP/Set_UE_posi_PPP_60s_60-75m_[10,90,50,10,90,90,10]user_7BS_.mat'
        UE_posi_filepath = 'UE_tra/0909_7BS/02Set_UE_posi_330user_edge_7BS.mat'
        posi_index = 'Set_UE_posi'

        '''额外干扰地图'''
        extra_itf_filepath = '7BS_outer_itf_map.mat'
        extra_itf_index = 'itf_map'

        '''大尺度信道预测模型'''
        model_name = 'scene0_large_h_DNN_0515'
        # model_name = 'DNN_0508'
        NN_path = 'DNNModel/large_h_predict/'+model_name+'/'+model_name+'.dat'
        normalize_para_filename = 'DNNModel/large_h_predict/'+model_name+'/normalize_para.npy'

        '''每个UE的RB数决策模型'''
        # actor_path = 'ReinfocementLearning/models/0709model/actor_1234567_0709_0'
        # actor_path = 'ReinforcementLearningV2/actor_1234567(0-30)_0726'
        actor_path = 'unsupervised0829/results/0905/actor_0905_200010_liji_duobu_0'
        actor_lr = 0.001
        actor_num_hidden_1 = 50
        actor_num_hidden_2 = 40
        actor_num_hidden_3 = 30
        actor_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        actor_config.gpu_options.allow_growth = True
        # actor = ActorNetwork(actor_lr, actor_num_hidden_1, actor_num_hidden_2, actor_num_hidden_3)

    '''仿真参数集'''
    PARAM_list = paraset_generator()

    def simulator_entry(PARAM_list, shadowFad_dB, UE_posi, extra_interf_map=None):
        if SimConfig.save_flag == 1:
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
            if PARAM.dynamic_nRB_per_UE and (PARAM.dynamic_nRB_scheme == 'new_model' or PARAM.dynamic_nRB_scheme == 'old_model'):
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
            _rate_arr, _UE_list, _BS_list, _UE_offline_dict, _rec_arr, _inter_arr, _RB_for_edge_ratio_arr, _UE_in_different_cell_arr, _GBR_UE_arr, _est_RL_state_dict = start_simulation(PARAM, Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map, NN, normalize_para, actor, sess, extra_interf_map)
            _end_time = time.time()

            '''一轮仿真结束，输出和保存信息'''
            print('Simulation of Parameter Set:{} Complete.'.format(i+1))
            print('Mean Rate:{:.2f} Mbps'.format(np.mean(_rate_arr[_rate_arr != 0])/1e6))
            # print('Center Cell Mean Rate:{:.2f} Mbps'.format(np.mean(_center_cell_rate_arr[_center_cell_rate_arr != 0]) / 1e6))
            print('Consumed Time:{:.2f}s\n'.format(_end_time - _start_time))
            rate_list.append(_rate_arr)
            if SimConfig.save_flag == 1:
                if not os.path.exists(SimConfig.root_path+'/{}'.format(i)):
                    os.makedirs(SimConfig.root_path+'/{}'.format(i))
                np.save(SimConfig.root_path + '/{}/rate_arr.npy'.format(i), _rate_arr)
                np.save(SimConfig.root_path + '/{}/UE_list.npy'.format(i), _UE_list)
                np.save(SimConfig.root_path + '/{}/BS_list.npy'.format(i), _BS_list)
                np.save(SimConfig.root_path + '/{}/UE_offline_dict.npy'.format(i), _UE_offline_dict)
                # np.save(SimConfig.root_path + '/{}/rec_arr.npy'.format(i), _rec_arr)
                # np.save(SimConfig.root_path + '/{}/inter_arr.npy'.format(i), _inter_arr)
                np.save(SimConfig.root_path + '/{}/UE_in_different_cell_arr.npy'.format(i), _UE_in_different_cell_arr)
                np.save(SimConfig.root_path + '/{}/est_RL_state_dict.npy'.format(i), _est_RL_state_dict)

                if PARAM.ICIC.flag and PARAM.ICIC.dynamic:
                    np.save(SimConfig.root_path + '/{}/RB_for_edge_ratio_arr.npy'.format(i), _RB_for_edge_ratio_arr)
                if PARAM.GBR_ratio != 0:
                    np.save(SimConfig.root_path + '/{}/GBR_UE_arr.npy'.format(i), _GBR_UE_arr)

            # HO_result_list.append(_HO_result)

        end_time = time.time()
        print('All Simulation Complete.')
        print('Total Consumed Time:{:.2f}s\n'.format(end_time - start_time))

        return


    np.random.seed(0)
    '''从文件读取阴影衰落'''
    shadowFad_dB = get_shadow_from_mat(SimConfig.shadow_filepath, SimConfig.shadow_index)
    # shadowFad_dB = np.zeros(shadowFad_dB.shape)  # 阴影置零，不考虑阴影

    '''从文件读取UE位置'''
    UE_posi = get_UE_posi_from_file(SimConfig.UE_posi_filepath, SimConfig.posi_index)
    # UE_posi = UE_posi[2, :, :]

    '''调整UE轨迹的数据为【行人，自行车，汽车】'''
    # if len(UE_posi.shape) != 3:
        # UE_posi = np.swapaxes(UE_posi, 0,1)
        # # up_v0_UE_posi = UE_posi[:30,:]
        # # up_v1_UE_posi = UE_posi[30:60,:]
        # # up_v2_UE_posi = UE_posi[60:90,:]
        # # down_v0_UE_posi = UE_posi[90:120,:]
        # # down_v1_UE_posi = UE_posi[120:150,:]
        # # down_v2_UE_posi = UE_posi[150:180,:]
        # # UE_posi = np.concatenate((up_v0_UE_posi,down_v0_UE_posi,up_v1_UE_posi, down_v1_UE_posi,up_v2_UE_posi,down_v2_UE_posi), axis=0)
        # UE_posi = np.reshape(UE_posi, (PARAM_list[0].ntype, -1, UE_posi.shape[1]))
        # UE_posi = np.swapaxes(UE_posi, 1, 2)

    UE_posi = process_posi_data(UE_posi, dis_threshold=1e6)

    # '''验证UE轨迹'''
    # fig, ax = plt.subplots()
    # # for i in range(8):
    # _UE_tra = UE_posi[1,:]
    # real_part = np.real(_UE_tra.tolist())
    # imag_part = np.imag(_UE_tra.tolist())
    # ax.scatter(real_part, imag_part)
    # plt.show()

    '''从文件读取额外干扰地图'''
    try:
        extra_interf_map = get_UE_posi_from_file(SimConfig.extra_itf_filepath, SimConfig.extra_itf_index)
    except:
        extra_interf_map = None

    '''进入仿真'''
    simulator_entry(PARAM_list, shadowFad_dB, UE_posi, extra_interf_map)

