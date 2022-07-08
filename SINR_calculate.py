'''
本模块包含SINR和数据率计算相关函数:
    get_receive_power
    get_interference
    calculate_SINR_dB
    update_SS_SINR
    user_rate
    ...
'''


from info_management import *
from precoding import ZF_precoding
import torch
from data_factory import search_object_form_list_by_no


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


def get_interference(PARAM, BS_list, UE_list, channel: InstantChannelMap, extra_interf_map=None):
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
            _inter_RB = RB_serv_arr[np.where(_BS.resource_map.RB_ocp_num[RB_serv_arr] != 0)[0]]
            _H_itf = H[:, _BS.no, :]
            for _RB in _inter_RB:
                _W = _BS.precoding_info[_RB].matrix

                _coe = _BS.precoding_info[_RB].coeffient  # 干扰基站预编码系数

                _H = H[:, _BS.no, _UE.no]  # 干扰基站与当前用户信道
                _itf = np.square(np.linalg.norm(np.dot(_H, np.sqrt(_coe) *_W)))
                interference_power[_UE.no, _RB] = interference_power[_UE.no, _RB] + _itf

    if extra_interf_map is not None:
        for _UE in UE_list:
            if _UE.serv_BS == -1: continue
            _BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
            if _UE.serv_BS == 2:
                probe = _UE.serv_BS
            RB_serv_arr = np.array([_UE.RB_Nt_ocp[i][0] for i in range(len(_UE.RB_Nt_ocp))])
            for _RB in RB_serv_arr:
                    UE_posi = _UE.posi
                    origin_x_point = PARAM.origin_x
                    origin_y_point = PARAM.origin_y
                    x_temp = int(np.ceil((np.real(UE_posi) - origin_x_point) / PARAM.posi_resolution))
                    y_temp = np.floor(np.ceil((np.imag(UE_posi) - origin_y_point) / PARAM.posi_resolution)).astype(int)
                    x_temp = np.min((extra_interf_map.shape[3] - 1, x_temp))
                    y_temp = np.min((extra_interf_map.shape[2] - 1, y_temp))
                    ICIC_RB_flag = 1*(_RB in _BS.resource_map.edge_RB_sorted_idx)
                    try:
                        _extra_itf = extra_interf_map[_UE.serv_BS, ICIC_RB_flag, y_temp, x_temp]
                    except:
                        raise Exception('Get extra interference Err')

                    interference_power[_UE.no, _RB] = interference_power[_UE.no, _RB] + _extra_itf


    return interference_power


def calculate_SINR_dB(receive_power, interference_power, noise):
    SINR = receive_power/(interference_power+noise)
    return 10*np.log10(SINR)


# def calculate_SS_SINR(receive_power, interference_power, noise):
#     # nRB = np.count_nonzero(receive_power, axis=1)
#     # mean_receive_power = np.sum(receive_power, axis=1) / nRB
#     # mean_interference_power = np.sum(interference_power, axis=1) / nRB
#     # SS_SINR = mean_receive_power / (mean_interference_power + noise)
#     interference_power_sum = np.zeros((150,))
#     receive_power_sum = np.sum(receive_power, axis=1)
#     for _UE_no in range(receive_power.shape[0]):
#         _interf_RB_idx = np.where(receive_power[_UE_no,:] != 0)
#         interference_power_sum[_UE_no] = np.sum(interference_power[_UE_no, _interf_RB_idx])
#     SS_SINR = receive_power_sum / (interference_power_sum + noise)
#     return SS_SINR

def update_SS_SINR(UE_list, BS_list, noise, mean_filter_length, after_HO=False):
    for _UE in UE_list:
        if _UE.no == 9:
            probe = _UE.no
        if not _UE.active: continue
        if after_HO and _UE.RL_state.filtered_SINR_dB != None: continue
        if _UE.state == 'unserved':
            BS_no = _UE.neighbour_BS[0]
            BS_L3_h = _UE.neighbour_BS_L3_h[0]
        else:
            BS_no = _UE.serv_BS
            BS_L3_h = _UE.serv_BS_L3_h

        _BS = search_object_form_list_by_no(BS_list, BS_no)
        Ptmax = _BS.Ptmax
        nNt = _BS.nNt
        nRB = len(_BS.center_RB_idx) + len(_BS.edge_RB_idx)
        try:
            K = np.min([_BS.MaxUE_per_RB, _BS.nUE_in_range])
        except:
            raise Exception('K is invalid!')

        AG = (nNt-K+1)/K  # ZF预编码下的AG
        rec_power = np.square(BS_L3_h) * Ptmax / nRB * AG
        neighbour_BS_L3_h = _UE.neighbour_BS_L3_h
        interf = np.sum(np.square(neighbour_BS_L3_h)) * Ptmax / nRB - np.square(BS_L3_h) * Ptmax / nRB
        SS_SINR = rec_power / (interf + noise)
        _UE.update_RL_state_by_SINR(SS_SINR, mean_filter_length)


def update_pred_SS_SINR(UE_list, noise, NN, normalize_para, pred_len):
    for _UE in UE_list:

        if not _UE.active: continue
        if len(_UE.RL_state.pred_SINR_dB) != 0:
            _UE.RL_state.pred_SINR_dB = _UE.RL_state.pred_SINR_dB[1:]
            continue

        x_large_h_dB = np.float32((10 * np.log10(_UE.all_BS_L3_h_record) - normalize_para['mean1']) / normalize_para[
            'sigma1'])

        if len(x_large_h_dB.shape) != 2:
            raise Exception('len(x_large_h_dB.shape) != 2')
        pred_large_h = []
        for _BS_no in range(x_large_h_dB.shape[1]):
            _x = x_large_h_dB[:, _BS_no]
            # x_serv = x_large_h[:, _UE.serv_BS]
            # x_target = x_large_h[:, _best_BS]
            _x = torch.tensor(_x)
            _h_pred_dB = np.array(NN.predict(_x).detach().cpu())
            _h_pred = 10 ** (_h_pred_dB / 10)  # (1,...)
            _h_pred = _h_pred[0][:pred_len]
            pred_large_h.append(_h_pred)
        pred_large_h = np.array(pred_large_h)  # (nBS, pred_len)

        if _UE.state == 'unserved':
            best_BS_no = _UE.neighbour_BS[0]
        else:
            best_BS_no = _UE.serv_BS

        pred_rec_power = np.square(pred_large_h[best_BS_no])
        pred_interf = np.sum(np.square(pred_large_h), axis=0) - pred_rec_power
        pred_SS_SINR = pred_rec_power / (pred_interf + noise)
        _UE.RL_state.pred_SINR_dB = 10*np.log10(pred_SS_SINR)



def user_rate(RB_width, SINR_dB, UE_list):
    SINR = 10**(SINR_dB/10)
    rate = RB_width * np.log2(1+SINR)
    rate_sum = np.sum(rate, axis=1)
    # for _UE in UE_list:
    #     if _UE.active and _UE.state == 'handovering' and _UE.HO_state.stage == 'HO_exec':
    #         rate_sum[_UE.no] = 1e-6
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
    large_fading = LargeScaleChannelMap(PARAM.Macro.nBS, PARAM.nUE)
    small_fading = SmallScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)
    instant_channel = InstantChannelMap(PARAM.Macro.nBS, PARAM.nUE, PARAM.Macro.nNt)
    serving_map = ServingMap(PARAM.Macro.nBS, PARAM.nUE)

    large_h = large_scale_channel(PARAM, Macro_BS_list, UE_posi[0, :], shadow)
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
