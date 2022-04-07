'''
本模块包含SINR和数据率计算相关函数:
    get_receive_power
    get_interference
    calculate_SINR_dB
    calculate_SS_SINR_dB
    user_rate
'''


from info_management import *
from precoding import ZF_precoding


def get_receive_power(BS_list, channel: InstantChannelMap, precoding_method=ZF_precoding):
    H = channel.map  # (nNt, nBS, nUE)
    nUE = H.shape[2]
    nRB = BS_list[0].nRB
    receive_power = np.zeros((nUE,nRB))
    for _BS in BS_list:
        _H = H[:, _BS.no, :]
        for _RB in range(_BS.nRB):
            _RB_resource = _BS.resource_map.map[_RB, :]
            _serv_UE = _RB_resource[np.where(_RB_resource != -1)].astype('int32')
            if len(_serv_UE) == 0: continue  # 若没有服务用户，跳过
            _coe = _BS.precoding_info[_RB].coeffient
            receive_power[_serv_UE,_RB] = receive_power[_serv_UE,_RB] + _coe

    return receive_power


def get_interference(BS_list, UE_list, channel: InstantChannelMap, precoding_method=ZF_precoding):
    H = channel.map  # (nNt, nBS, nUE)
    nUE = H.shape[2]
    nRB = BS_list[0].nRB
    interference_power = np.zeros((nUE,nRB))
    for _UE in UE_list:
        if _UE.serv_BS == -1: continue
        '''服务频段'''
        RB_serv_arr = np.array([_UE.RB_Nt_ocp[i][0] for i in range(len(_UE.RB_Nt_ocp))])
        for _BS in BS_list:
            if _UE.serv_BS == _BS.no: continue  # 如果是服务基站，不计算干扰
            if _BS.no not in _UE.neighbour_BS: continue  # 如果不在邻基站列表内，忽略干扰
            '''找到有干扰的频段'''
            _inter_RB = np.where(_BS.resource_map.RB_ocp_num[RB_serv_arr] != 0)[0]
            _H_itf = H[:, _BS.no, :]
            for _RB in _inter_RB:
                _W = _BS.precoding_info[_RB].matrix
                _coe = _BS.precoding_info[_RB].coeffient  # 干扰基站预编码系数

                _H = H[:, _BS.no, _UE.no]  # 干扰基站与当前用户信道
                _itf = np.sum(_coe * np.square(np.linalg.norm(np.dot(_H, _W))))
                interference_power[_UE.no, _RB] = interference_power[_UE.no, _RB] + _itf


    return interference_power


def calculate_SINR_dB(receive_power, interference_power, noise):
    SINR = receive_power/(interference_power+noise)
    return 10*np.log10(SINR)


def calculate_SS_SINR_dB(receive_power, interference_power, noise):
    nRB = np.count_nonzero(receive_power, axis=1)
    mean_receive_power = np.sum(receive_power, axis=1)/nRB
    mean_interference_power = np.sum(interference_power, axis=1)/nRB
    SS_SINR = mean_receive_power/(mean_interference_power+noise)
    return 10*np.log10(SS_SINR)


def user_rate(RB_width, SINR_dB):
    SINR = 10**(SINR_dB/10)
    rate = RB_width * np.log2(1+SINR)
    rate_sum = np.sum(rate, axis=1)
    return rate_sum


def calculate_SNR_dB(receive_power, noise):
    SNR = receive_power/(noise)
    return 10*np.log10(SNR)

if __name__ == '__main__':
    '''
    用于测试
    '''
    from simulator import Parameter
    from network_deployment import cellStructPPP
    from user_mobility import get_UE_posi_from_mat
    from channel_fading import *
    from radio_access import access_init

    np.random.seed(0)
    PARAM = Parameter()
    filepath = 'shadowFad_dB1.mat'
    index = 'shadowFad_dB'
    shadowFad_dB = get_shadow_from_mat(filepath, index)
    # print(shadowFad_dB[0][1])
    filepath = 'Set_UE_posi_60s_250user_1to2_new.mat'
    index = 'Set_UE_posi'
    UE_posi = get_UE_posi_from_mat(filepath, index)

    Macro_Posi, Micro_Posi, nMicro = cellStructPPP(PARAM.nCell, PARAM.Dist, PARAM.Micro.nBS_avg)
    Macro_BS_list = []
    print("Macro Ptmax:", PARAM.Macro.Ptmax)
    for i in range(PARAM.Macro.nBS):
        Macro_BS_list.append(BS(i, 'Macro', PARAM.Macro.nNt, PARAM.nRB,PARAM.Macro.Ptmax, Macro_Posi[i], True, PARAM.Macro.MaxUE_per_RB))

    UE_list = []
    # random_UE_idx = np.random.choice(len(UE_posi[0]),PARAM.nUE,replace=False)
    for i in range(PARAM.nUE):
        UE_list.append(UE(i, UE_posi[0,i], True))

    shadow = ShadowMap(shadowFad_dB[0])
    large_fading = LargeScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE)
    small_fading = SmallScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)
    instant_channel = InstantChannelMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)
    serving_map = ServingMap(PARAM.Macro.nBS, PARAM.nUE)

    large_h = large_scale_fading(PARAM, Macro_BS_list, UE_posi[0, :], shadow)
    large_fading.update(large_h)
    # print(_large_h[2, 4:6], large_fading.map[2, 4:6])  # 看更新后一不一致
    small_h = small_scale_fading(PARAM.nUE, len(Macro_BS_list), PARAM.Macro.nNt)
    small_fading.update(small_h)
    # print('small_h shape:', small_h.shape)
    instant_channel.calculate_by_fading(large_fading, small_fading)

    result = access_init(PARAM, Macro_BS_list, UE_list, large_fading, serving_map)
    # print(result)

    rec_P = get_receive_power(Macro_BS_list, instant_channel)
    inter_P = get_interference(Macro_BS_list, UE_list, instant_channel)
    # print(PARAM.sigma2, PARAM.sigma_c)
    SINR_dB = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma2)
    SINR_dB_c = calculate_SINR_dB(rec_P, inter_P, PARAM.sigma_c)
    SNR_dB = calculate_SNR_dB(rec_P, PARAM.sigma2)
    UE_rate = user_rate(PARAM.MLB.RB, SINR_dB)
    UE_rate_c = user_rate(PARAM.MLB.RB, SINR_dB_c)
    print('初次接入: 采用sigma2 平均数据率：{:.2f} Mbps'.format(np.mean(UE_rate)/1e6))
    print('采用sigma_c 平均数据率：{:.2f} Mbps'.format(np.mean(UE_rate_c)/1e6))
    # print(UE_list[0].serv_BS,UE_list[240].serv_BS)
