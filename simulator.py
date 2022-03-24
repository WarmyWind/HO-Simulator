'''
本模块为仿真器入口
'''


import numpy as np
import time
from para_init import *
from network_deployment import cellStructPPP
from user_mobility import get_UE_posi_from_mat
from channel_fading import *
from radio_access import access_init
from SINR_calculate import *
from handover_procedure import handover_criteria_eval

class SimConfig:  # 仿真参数
    plot_flag = 1  # 是否绘图
    nDrop = 60  # 时间步进长度，相当于60个循环

def start_simulation(PARAM, BS_list, UE_list, shadow, large_fading:LargeScaleFadingMap, small_fading:SmallScaleFadingMap,
                     instant_channel:InstantChannelMap, serving_map:ServingMap):
    '''初始接入'''
    _ = access_init(PARAM, BS_list, UE_list, large_fading, serving_map)
    rec_P = get_receive_power(BS_list, instant_channel)
    inter_P = get_interference(BS_list, UE_list, instant_channel)
    SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma2)
    UE_rate = user_rate(PARAM.MLB.RB, SINR_dB)
    # print(np.mean(UE_rate))

    rate_list = [UE_rate]
    '''开始步进时长仿真'''
    for drop_idx in range(1, SimConfig.nDrop):
        '''更新UE位置'''
        for _UE in UE_list:
            _UE.update_posi(UE_posi[drop_idx, _UE.no])

        '''更新小尺度信道信息'''
        small_h = small_scale_fading(PARAM.nUE, len(BS_list), PARAM.Macro.nNt)
        small_fading.update(small_h)

        '''更新大尺度信道信息'''
        large_h = large_scale_fading(PARAM, BS_list, UE_posi[drop_idx, :], shadow)
        large_fading.update(large_h)

        '''更新瞬时信道信息'''
        instant_channel.calculate_by_fading(large_fading, small_fading)

        '''开始HO eval'''
        HOM = 3  # dB
        TTT = 4
        handover_criteria_eval(PARAM, UE_list, BS_list, large_fading, HOM, TTT, serving_map, 'avg')

        '''统计性能'''
        rec_P = get_receive_power(BS_list, instant_channel)
        inter_P = get_interference(BS_list, UE_list, instant_channel)
        # print(PARAM.sigma2, PARAM.sigma_c)
        SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma2)
        # SNR_dB = calculate_SNR_dB(rec_P, PARAM.sigma2)
        UE_rate = user_rate(PARAM.MLB.RB, SINR_dB)
        rate_list.append(UE_rate)

    # print(np.min(mean_rate), np.max(mean_rate))
    HO_success = 0
    HO_failure = 0
    failure_repeat_UE_num = 0
    for _UE in UE_list:
        HO_success = HO_success + _UE.HO_state.success_count
        HO_failure = HO_failure + _UE.HO_state.failure_count
        if _UE.HO_state.failure_count > 1:
            failure_repeat_UE_num = failure_repeat_UE_num + 1
    print('HO results: HO success count: {}, HO failure count: {}, failure_repeat_UE_num:{}'.format(HO_success, HO_failure, failure_repeat_UE_num))
    return np.array(rate_list), np.array([HO_success, HO_failure, failure_repeat_UE_num])

def init_all(PARAM, Macro_Posi, UE_posi, shadowFad_dB):
    '''初始化所有对象'''
    '''创建BS对象，并加入列表'''
    Macro_BS_list = []
    # print("Macro Ptmax:", PARAM.Macro.Ptmax)
    for i in range(PARAM.Macro.nBS):
        Macro_BS_list.append(BS(i, 'Macro', PARAM.Macro.nNt, PARAM.nRB,PARAM.Macro.Ptmax, Macro_Posi[i], True, PARAM.Macro.MaxUE_per_RB))

    '''创建UE对象，并加入列表'''
    UE_list = []
    # random_UE_idx = np.random.choice(len(UE_posi[0]),PARAM.nUE,replace=False)
    for i in range(PARAM.nUE):
        UE_list.append(UE(i, UE_posi[0,i], True))

    '''初始化信道、服务信息'''
    shadow = ShadowMap(shadowFad_dB[0])
    large_fading = LargeScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE)
    small_fading = SmallScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)
    instant_channel = InstantChannelMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)

    large_h = large_scale_fading(PARAM, Macro_BS_list, UE_posi[0, :], shadow)
    large_fading.update(large_h)
    small_h = small_scale_fading(PARAM.nUE, len(Macro_BS_list), PARAM.Macro.nNt)
    small_fading.update(small_h)
    instant_channel.calculate_by_fading(large_fading, small_fading)
    serving_map = ServingMap(PARAM.Macro.nBS, PARAM.nUE)

    return Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map

if __name__ == '__main__':
    PARAM = Parameter()
    # print(PARAM.nCell, PARAM.Macro.Ptmax, PARAM.pathloss.Macro.dFactordB, PARAM.MLB.RB)
    np.random.seed(0)
    '''从文件读取阴影衰落'''
    filepath = 'shadowFad_dB1.mat'
    index = 'shadowFad_dB'
    shadowFad_dB = get_shadow_from_mat(filepath, index)
    # print(shadowFad_dB[0][1])

    '''从文件读取UE位置'''
    filepath = 'Set_UE_posi_60s_250user_1to2_new.mat'
    index = 'Set_UE_posi'
    UE_posi = get_UE_posi_from_mat(filepath, index)

    '''生成BS位置'''
    Macro_Posi, _, _ = cellStructPPP(PARAM.nCell, PARAM.Dist, PARAM.Micro.nBS_avg)


    '''开始仿真'''
    RB_per_UE_list = [4,3,2]
    rate_list = []
    HO_result_list = []
    start_time = time.time()
    print('Simulation Start.\n')
    for i in range(len(RB_per_UE_list)):
        print('Simulation of Parameter Set:{} Start.'.format(i+1))
        PARAM.RB_per_UE = RB_per_UE_list[i]
        '''初始化对象'''
        Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map= init_all(PARAM, Macro_Posi, UE_posi, shadowFad_dB)
        _start_time = time.time()
        _rate_arr, _HO_result = start_simulation(PARAM, Macro_BS_list, UE_list, shadow, large_fading, small_fading, instant_channel, serving_map)
        _end_time = time.time()
        print('Simulation of Parameter Set:{} Complete.'.format(i+1))
        print('Comsumed Time:{:.2f}s\n'.format(_end_time - _start_time))
        rate_list.append(_rate_arr)
        HO_result_list.append(_HO_result)

    end_time = time.time()
    print('All Simulation Complete.')
    print('Total comsumed Time:{:.2f}s\n'.format(end_time - start_time))

    if SimConfig.plot_flag == 1:
        from visualization import plot_cdf, plot_bar
        rate_data = rate_list
        label_list = ['RB_per_UE={}'.format(n) for n in RB_per_UE_list]
        plot_cdf(rate_data, 'bit rate', 'cdf', label_list)

        HO_result = np.array(HO_result_list).transpose()
        HO_result = [HO_result[i] for i in range(len(HO_result))]
        para_list = ['RB={}'.format(n) for n in RB_per_UE_list]
        label_list = ['Success', 'Failure', 'Num of Failure Repeat UE']
        plot_bar(HO_result, 'Parameter Set', 'HO result', para_list, label_list)