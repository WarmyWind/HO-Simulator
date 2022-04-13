'''
本模块为仿真器入口
'''



import numpy as np
import time
import os
from para_init import *
from network_deployment import cellStructPPP, road_cell_struct
from user_mobility import *
from channel_fading import *
from radio_access import access_init, find_and_update_neighbour_BS
from channel_measurement import update_serv_BS_L3_h
from SINR_calculate import *
from handover_procedure import handover_criteria_eval
from utils import *
import warnings
warnings.filterwarnings('ignore')

def start_simulation(PARAM, BS_list, UE_list, shadow, large_fading:LargeScaleFadingMap, small_fading:SmallScaleFadingMap,
                     instant_channel:InstantChannelMap, serving_map:ServingMap):
    '''开始仿真'''

    '''更新UE的邻基站'''
    find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading, instant_channel, PARAM.L3_coe)

    '''初始接入'''
    _ = access_init(PARAM, BS_list, UE_list, instant_channel, serving_map)

    '''更新UE的服务基站L3测量'''
    update_serv_BS_L3_h(UE_list, instant_channel, PARAM.L3_coe)

    '''更新预编码信息'''
    for _BS in BS_list:
        _BS.update_precoding_matrix(instant_channel, ZF_precoding)

    '''接入时SINR和速率'''
    rec_P = get_receive_power(BS_list, instant_channel)
    inter_P = get_interference(BS_list, UE_list, instant_channel)
    SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma_c)
    UE_rate = user_rate(PARAM.MLB.RB, SINR_dB)
    # print(np.mean(UE_rate))
    rate_list = [UE_rate]

    '''接入时RL state'''
    # SS_SINR_list = []
    SS_SINR = calculate_SS_SINR(rec_P, inter_P, PARAM.sigma_c)
    # SS_SINR_list.append(SS_SINR)
    for _UE in UE_list:
        if _UE.active:
            _UE.update_RL_state_by_SINR(SS_SINR[_UE.no], PARAM.L1_filter_length)

    '''开始步进时长仿真'''
    for drop_idx in range(1, SimConfig.nDrop):
        '''更新UE位置'''
        if drop_idx % PARAM.posi_resolution == 0:
            _posi_idx = int(drop_idx // PARAM.posi_resolution)
            for _UE in UE_list:
                if isinstance(UE_posi, list):
                    _UE_posi = UE_posi[_UE.type][_posi_idx, _UE.type_no]
                    _UE.update_posi(_UE_posi)
                elif len(UE_posi.shape) == 2:
                    _UE_posi = UE_posi[_posi_idx, _UE.type_no]
                    _UE.update_posi(_UE_posi)
                elif len(UE_posi.shape) == 3:
                    _UE_posi = UE_posi[_UE.type, _posi_idx, _UE.type_no]
                    _UE.update_posi(_UE_posi)

        '''更新小尺度信道信息'''
        small_h = small_scale_fading(PARAM.nUE, len(BS_list), PARAM.Macro.nNt)
        small_fading.update(small_h)

        '''更新大尺度信道信息'''
        large_h = large_scale_fading(PARAM, BS_list, UE_list, shadow)
        large_fading.update(large_h)

        '''更新瞬时信道信息'''
        instant_channel.calculate_by_fading(large_fading, small_fading)

        '''新出现的活动UE进行接入，停止活动的UE断开'''
        for _UE in UE_list:
            if not _UE.active:
                if _UE.serv_BS != -1:
                    _serv_BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
                    _serv_BS.unserve_UE(_UE, serving_map)  # 断开原服务，释放资源
            else:
                if _UE.serv_BS == -1:
                    _ = access_init(PARAM, BS_list, [_UE], instant_channel, serving_map)


        '''更新UE的服务基站L3测量'''
        update_serv_BS_L3_h(UE_list, instant_channel, PARAM.L3_coe)

        '''更新UE的邻基站及其的L3测量'''
        find_and_update_neighbour_BS(BS_list, UE_list, PARAM.num_neibour_BS_of_UE, large_fading, instant_channel, PARAM.L3_coe)


        '''开始HO eval'''
        # HOM = 3  # dB
        # TTT = 32
        measure_criteria = 'L3'
        handover_criteria_eval(PARAM, UE_list, BS_list, large_fading, instant_channel, PARAM.HOM, PARAM.TTT,
                                serving_map, measure_criteria)

        '''更新预编码信息'''
        for _BS in BS_list:
            _BS.update_precoding_matrix(instant_channel, ZF_precoding)

        '''统计性能'''
        rec_P = get_receive_power(BS_list, instant_channel)
        inter_P = get_interference(BS_list, UE_list, instant_channel)
        SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma_c)
        # SNR_dB = calculate_SNR_dB(rec_P, PARAM.sigma2)
        UE_rate = user_rate(PARAM.MLB.RB, SINR_dB)
        rate_list.append(UE_rate)

        '''更新RL state'''
        SS_SINR = calculate_SS_SINR(rec_P, inter_P, PARAM.sigma_c)
        # SS_SINR_list.append(SS_SINR)
        # if len(SS_SINR) < PARAM.L1_filter_length:
        #     _SS_SINR = np.mean(np.array(SS_SINR_list), axis=0)
        # else:
        #     _SS_SINR = np.mean(np.array(SS_SINR_list)[-PARAM.L1_filter_length:, :], axis=0)
        for _UE in UE_list:
            if _UE.active:
                # if _UE.RL_state.active == False:
                #     print('there')
                _UE.update_RL_state_by_SINR(SS_SINR[_UE.no], PARAM.L1_filter_length)

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

    return np.array(rate_list), UE_list

def init_all(PARAM, Macro_Posi, UE_posi, shadowFad_dB):
    '''初始化所有对象'''
    '''创建BS对象，并加入列表'''
    Macro_BS_list = []
    # print("Macro Ptmax:", PARAM.Macro.Ptmax)
    for i in range(PARAM.Macro.nBS):
        Macro_BS_list.append(BS(i, 'Macro', PARAM.Macro.nNt, PARAM.nRB, PARAM.Macro.Ptmax, Macro_Posi[i], True, PARAM.Macro.MaxUE_per_RB))

    '''创建UE对象，并加入列表'''
    UE_list = []
    # random_UE_idx = np.random.choice(len(UE_posi[0]),PARAM.nUE,replace=False)

    if isinstance(UE_posi, list):
        for i in range(len(UE_posi)):
            _UE_posi_arr = UE_posi[i]
            for _UE_no in range(PARAM.nUE_per_type):
                _UE_posi = _UE_posi_arr[0, _UE_no]
                if _UE_posi != None:
                    _active = True
                else:
                    _active = False
                UE_list.append(UE(i*PARAM.nUE_per_type+_UE_no, _UE_no, _UE_posi, i, active=_active))
    elif len(UE_posi.shape) == 2:
        for _UE_no in range(PARAM.nUE):
            _UE_posi = UE_posi[0, _UE_no]
            if _UE_posi != None:
                _active = True
            else:
                _active = False
            UE_list.append(UE(_UE_no, _UE_no, _UE_posi, active=_active))
    elif len(UE_posi.shape) == 3:
        for i in range(PARAM.nUE_per_type):
            for _UE_no in range(UE_posi.shape[2]):
                _UE_posi = UE_posi[i, 0, _UE_no]
                if _UE_posi != None:
                    _active = True
                else:
                    _active = False
                UE_list.append(UE(i*PARAM.nUE_per_type+_UE_no, _UE_no, _UE_posi, i, active=_active))


    '''初始化信道、服务信息'''
    shadow = ShadowMap(shadowFad_dB)
    large_fading = LargeScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE)
    small_fading = SmallScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)
    instant_channel = InstantChannelMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)

    large_h = large_scale_fading(PARAM, Macro_BS_list, UE_list, shadow)
    large_fading.update(large_h)
    small_h = small_scale_fading(PARAM.nUE, len(Macro_BS_list), PARAM.Macro.nNt)
    small_fading.update(small_h)
    instant_channel.calculate_by_fading(large_fading, small_fading)
    serving_map = ServingMap(PARAM.Macro.nBS, PARAM.nUE)

    return Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map

if __name__ == '__main__':
    class SimConfig:  # 仿真参数
        plot_flag = 0  # 是否绘图
        save_flag = 1  # 是否保存结果
        root_path = 'result/0414_2'
        nDrop = 10000  # 时间步进长度

    def simulator_entry(PARAM_list, shadowFad_dB, UE_posi):
        if SimConfig.save_flag == 1:
            # _path = SimConfig.root_path+'/PARAM_list.npy'
            if not os.path.exists(SimConfig.root_path):
                os.makedirs(SimConfig.root_path)
            np.save(SimConfig.root_path+'/PARAM_list.npy', PARAM_list)

        _PARAM = PARAM_list[0]
        '''生成BS位置'''
        # Macro_Posi, _, _ = cellStructPPP(PARAM.nCell, PARAM.Dist, PARAM.Micro.nBS_avg)
        Macro_Posi = road_cell_struct(_PARAM.nCell, _PARAM.Dist)

        '''开始仿真'''
        rate_list = []
        HO_result_list = []
        start_time = time.time()
        print('Simulation Start.\n')
        print('Important Parameters:')
        print('Sigma: sigma_c\n')

        for i in range(len(PARAM_list)):
            PARAM = PARAM_list[i]
            print('Simulation of Parameter Set:{} Start.'.format(i+1))

            Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map = init_all(PARAM, Macro_Posi, UE_posi, shadowFad_dB)
            _start_time = time.time()
            _rate_arr, _UE_list = start_simulation(PARAM, Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map)
            _end_time = time.time()
            print('Simulation of Parameter Set:{} Complete.'.format(i+1))
            print('Mean Rate:{:.2f} Mbps'.format(np.mean(_rate_arr[_rate_arr != 0])/1e6))
            print('Consumed Time:{:.2f}s\n'.format(_end_time - _start_time))
            rate_list.append(_rate_arr)
            if SimConfig.save_flag == 1:
                if not os.path.exists(SimConfig.root_path+'/{}'.format(i)):
                    os.makedirs(SimConfig.root_path+'/{}'.format(i))
                np.save(SimConfig.root_path+'/{}/rate_arr.npy'.format(i), _rate_arr)
                np.save(SimConfig.root_path+'/{}/UE_list.npy'.format(i), _UE_list)
            # HO_result_list.append(_HO_result)

        end_time = time.time()
        print('All Simulation Complete.')
        print('Total Consumed Time:{:.2f}s\n'.format(end_time - start_time))

        # if SimConfig.plot_flag == 1:
        #     from visualization import plot_cdf, plot_bar, plot_rate_map
        #     rate_data = rate_list
        #     label_list = ['RB_per_UE={}'.format(n) for n in RB_per_UE_list]
        #     plot_cdf(rate_data, 'bit rate', 'cdf', label_list)
        #
        #     HO_result = np.array(HO_result_list).transpose()
        #     HO_result = [HO_result[i] for i in range(len(HO_result))]
        #     para_list = ['RB={}'.format(n) for n in RB_per_UE_list]
        #     label_list = ['Success', 'Failure', 'Num of Failure Repeat UE']
        #     plot_bar(HO_result, 'Parameter Set', 'HO result', para_list, label_list)
        #
        #
        #     if len(UE_posi.shape) == 3:
        #         trans_UE_posi = np.zeros(UE_posi.shape[1], PARAM.nUE)
        #         _type = 0
        #         _UE_no = -1
        #         for nDrop in range(UE_posi.shape[1]):
        #             for _UE_type_no in range(UE_posi.shape[2]):
        #                 if _UE_type_no >= PARAM.nUE_per_type:
        #                     _type += 1
        #                     continue
        #                 _UE_no += 1
        #                 trans_UE_posi[nDrop, _UE_no] = UE_posi[_type, nDrop, _UE_type_no]

            # plot_rate_map(Macro_Posi, UE_posi, rate_data, para_list)
        return


    PARAM_list = []
    PARAM = Parameter()
    # PARAM_list.append(PARAM)
    HOM_list = [-4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6]
    # TTT_list = [8, 16, 24, 32] #  [48, 64, 96, 128]
    TTT_list = [48, 64, 96, 128]
    for _HOM in HOM_list:
        PARAM.HOM = _HOM
        for _TTT in TTT_list:
            PARAM.TTT = _TTT
            PARAM_list.append(PARAM)



    np.random.seed(0)
    '''从文件读取阴影衰落'''
    filepath = 'shadowFad_dB_8sigma.mat'
    index = 'shadowFad_dB'
    shadowFad_dB = get_shadow_from_mat(filepath, index)
    # probe = shadowFad_dB[0][1]

    '''从文件读取UE位置'''
    # filepath = 'Set_UE_posi_60s_250user_1to2_new1.mat'
    filepath = ['Set_UE_posi_100s_500user_v{}.mat'.format(i + 1) for i in range(3)]
    index = 'Set_UE_posi'
    UE_posi = get_UE_posi_from_mat(filepath, index)
    # UE_posi = UE_posi[2, :, :]
    UE_posi = process_posi_data(UE_posi)

    simulator_entry(PARAM_list, shadowFad_dB, UE_posi)

