'''
本模块包含SINR和数据率计算相关函数:
    get_receive_power
    get_interference
    calculate_SINR_dB
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
            '''这里的功率分配简单的以RB上的用户数作为系数'''
            _Pt_ratio = _BS.resource_map.RB_ocp_num[_RB] / np.sum(_BS.resource_map.RB_ocp_num)  # 占用的功率比例
            _, _cof = precoding_method(_H[:, _serv_UE].T, _BS.Ptmax * _Pt_ratio)
            receive_power[_serv_UE,_RB] = receive_power[_serv_UE,_RB] + _BS.nRB * _cof / _BS.resource_map.RB_ocp_num[_RB]

    return receive_power


def get_interference(BS_list, UE_list, channel: InstantChannelMap):
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
            '''找到有干扰的频段'''
            _inter_RB = np.where(_BS.resource_map.RB_ocp_num[RB_serv_arr] != 0)[0]
            for _RB in _inter_RB:
                '''找到有干扰的天线'''
                _inter_Nt = _BS.resource_map.RB_ocp[_RB].astype('int32')

                '''这里的功率分配简单的以RB上的用户数作为系数'''
                _Pt_ratio = _BS.resource_map.RB_ocp_num[_RB] / np.sum(_BS.resource_map.RB_ocp_num)  # 占用的功率比例
                interference_power[_UE.no, _RB] = interference_power[_UE.no, _RB] \
                            + _BS.nRB * _BS.Ptmax * _Pt_ratio * np.square(np.linalg.norm(H[_inter_Nt, _BS.no, _UE.no]))

    return interference_power


def calculate_SINR_dB(receive_power, interference_power, noise):
    SINR = receive_power/(interference_power+noise)
    return 10*np.log10(SINR)


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
    print(np.mean(UE_rate))
    # print(UE_list[0].serv_BS,UE_list[240].serv_BS)