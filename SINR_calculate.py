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
    H = channel.map  # (nRB, nNt, nBS, nUE)
    nUE = H.shape[3]
    nRB = BS_list[0].nRB
    receive_power = np.zeros((nUE, nRB))
    for _BS in BS_list:
        _H = H[:, :, _BS.no, :]
        for _RB in range(_BS.nRB):
            _RB_resource = _BS.resource_map.map[_RB, :]
            _serv_UE = _RB_resource[np.where(_RB_resource != -1)].astype('int32')
            if len(_serv_UE) == 0: continue  # 若没有服务用户，跳过
            _coe = _BS.precoding_info[_RB].coeffient
            # for _UE_no in _serv_UE:
            try:
                receive_power[_serv_UE,_RB] = receive_power[_serv_UE,_RB] + _coe
            except:
                raise Exception('Error')

    return receive_power


def get_interference(PARAM, BS_list, UE_list, channel: InstantChannelMap, extra_interf_map=None):
    H = channel.map  # (nRB, nNt, nBS, nUE)
    nUE = H.shape[3]
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
            _H_itf = H[:, :, _BS.no, :]
            for _RB in _inter_RB:
                _W = _BS.precoding_info[_RB].matrix

                _coe = _BS.precoding_info[_RB].coeffient  # 干扰基站预编码系数

                _H = H[_RB, :, _BS.no, _UE.no]  # 干扰基站与当前用户信道
                _itf = np.square(np.linalg.norm(np.dot(_H, np.sqrt(_coe) *_W)))
                interference_power[_UE.no, _RB] = interference_power[_UE.no, _RB] + _itf

    if extra_interf_map is not None:
        for _UE in UE_list:
            if _UE.serv_BS == -1: continue
            _BS = search_object_form_list_by_no(BS_list, _UE.serv_BS)
            # if _UE.serv_BS == 2:
            #     probe = _UE.serv_BS
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

def update_SS_SINR(PARAM, UE_list, BS_list, after_HO=False, extra_interf_map=None):
    noise = PARAM.sigma2
    mean_filter_length = PARAM.filter_length_for_SINR

    for _UE in UE_list:
        # if _UE.no == 37:
        #     probe = _UE.no
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
        nRB = PARAM.nRB

        try:
            K = np.min([_BS.opt_UE_per_RB, _BS.nUE_in_range])
        except:
            raise Exception('K is invalid!')

        AG = (nNt-K+1)/K  # ZF预编码下的AG
        rec_power = np.square(BS_L3_h) * Ptmax / nRB * AG
        neighbour_BS_L3_h = _UE.neighbour_BS_L3_h
        interf = np.sum(np.square(neighbour_BS_L3_h)) * Ptmax / nRB - np.square(BS_L3_h) * Ptmax / nRB
        if interf < 0:
            interf = 0


        ICIC_itf = 0
        if _BS.ICIC_group != -1:
            for neighbour_BS_list_idx in range(len(_UE.neighbour_BS)):
                _BS_no = _UE.neighbour_BS[neighbour_BS_list_idx]
                _neighbour_BS = search_object_form_list_by_no(BS_list, _BS_no)
                if _neighbour_BS.no != _BS.no and _neighbour_BS.ICIC_group == _BS.ICIC_group:
                    ICIC_itf = ICIC_itf + (np.square(neighbour_BS_L3_h[neighbour_BS_list_idx])) * Ptmax / nRB
        else:
            ICIC_itf = interf

        # 添加外圈干扰
        try:
            UE_posi = _UE.posi
            origin_x_point = PARAM.origin_x
            origin_y_point = PARAM.origin_y
            x_temp = int(np.ceil((np.real(UE_posi) - origin_x_point) / PARAM.posi_resolution))
            y_temp = np.floor(np.ceil((np.imag(UE_posi) - origin_y_point) / PARAM.posi_resolution)).astype(int)
            x_temp = np.min((extra_interf_map.shape[3] - 1, x_temp))
            y_temp = np.min((extra_interf_map.shape[2] - 1, y_temp))
            extra_ICIC_itf = extra_interf_map[_UE.serv_BS, 1, y_temp, x_temp]
            extra_non_ICIC_itf = extra_interf_map[_UE.serv_BS, 0, y_temp, x_temp]
        except:
            # extra_itf = 0
            raise Exception('Get extra interference Err')


        ICIC_SINR = rec_power / (ICIC_itf + extra_ICIC_itf + noise)

        SS_SINR = rec_power / (interf + extra_non_ICIC_itf + noise)
        _UE.update_RL_state_by_SINR(SS_SINR, mean_filter_length)


        _UE.RL_state.estimated_rec_power = rec_power
        _UE.RL_state.estimated_itf_power = interf + extra_non_ICIC_itf
        _UE.RL_state.estimated_ICIC_itf_power = ICIC_itf + extra_ICIC_itf
        _UE.RL_state.estimated_ICIC_SINR_dB = 10*np.log10(ICIC_SINR)


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


