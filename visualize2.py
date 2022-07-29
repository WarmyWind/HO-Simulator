from visualization import *


def find_UE_idx(UE_posi, current_frame, UE_no):
    _no = 0
    idx = 0
    if len(UE_posi) !=3:
        _UE_posi_list = UE_posi[current_frame, :]
        for _UE_posi in _UE_posi_list:
            if _no == UE_no:
                return idx

            if _UE_posi is not None:
                idx = idx + 1
            _no = _no + 1
    else:
        for i in range(len(UE_posi)):
            _UE_posi_list = UE_posi[i][current_frame,:]
            for _UE_posi in _UE_posi_list:
                if _no == UE_no:
                    return idx

                if _UE_posi is not None:
                    idx = idx + 1
                _no = _no+1

def count_offline_rate(UE_offline_dict):
    offline_rate_temp_list = []
    if len(UE_offline_dict) == 5:
        center_UE_offline_arr = np.array(UE_offline_dict['center_UE_offline'])
        edge_UE_offline_arr = np.array(UE_offline_dict['edge_UE_offline'])
        for i in range(len(center_UE_offline_arr)):
            offline_nUE = len(center_UE_offline_arr[i]) + len(edge_UE_offline_arr[i])
            _offline_rate = offline_nUE / num_UE
            offline_rate_temp_list.append(_offline_rate)
        return np.mean(offline_rate_temp_list)


    else:
        UE_offline_arr = np.array(UE_offline_dict['UE_offline'])
        for i in range(len(UE_offline_arr)):
            offline_nUE = len(UE_offline_arr[i])
            _offline_rate = offline_nUE / num_UE
            offline_rate_temp_list.append(_offline_rate)
        return np.mean(offline_rate_temp_list)

if __name__ == '__main__':
    from simulator import *
    from channel_fading import get_shadow_from_mat
    from user_mobility import get_UE_posi_from_file
    from network_deployment import cellStructPPP

    PARAM = Parameter()
    '''从文件读取UE位置'''
    # UE_posi_filepath = 'UE_tra/0714_7road_7BS/Set_UE_posi_60s_550user_7BS.mat'
    UE_posi_filepath = 'UE_tra/0714_7road_7BS/Set_UE_posi_60s_330user_7BS_V123.mat'
    # UE_posi_filepath = 'UE_tra/0627_7road_7BS/Set_UE_posi_60s_330user_7BS_V123.mat'
    # UE_posi_filepath = 'UE_tra/0721_7BS_PPP/Set_UE_posi_PPP_60s_[80,20,50,70,30,30,50]user_7BS_.mat'
    # UE_posi_filepath = 'UE_tra/0721_7BS_PPP/Set_UE_posi_PPP_60s_60-75m_[80,20,50,70,30,30,50]user_7BS_.mat'
    # UE_posi_filepath = 'UE_tra/0726_7BS_withstaticUE/0727edge2.mat'
    index = 'Set_UE_posi'
    UE_posi = get_UE_posi_from_file(UE_posi_filepath, index)
    # UE_posi = UE_posi[2, :, :]
    if len(UE_posi.shape) != 3:
        UE_posi = np.swapaxes(UE_posi, 0, 1)
        # up_v0_UE_posi = UE_posi[:30, :]
        # up_v1_UE_posi = UE_posi[30:60, :]
        # up_v2_UE_posi = UE_posi[60:90, :]
        # down_v0_UE_posi = UE_posi[90:120, :]
        # down_v1_UE_posi = UE_posi[120:150, :]
        # down_v2_UE_posi = UE_posi[150:180, :]
        # UE_posi = np.concatenate(
        #     (up_v0_UE_posi, down_v0_UE_posi, up_v1_UE_posi, down_v1_UE_posi, up_v2_UE_posi, down_v2_UE_posi), axis=0)
        UE_posi = np.reshape(UE_posi, (PARAM.ntype, -1, UE_posi.shape[1]))
        UE_posi = np.swapaxes(UE_posi, 1, 2)

    UE_posi = process_posi_data(UE_posi, dis_threshold=1e6)

    # '''验证UE轨迹'''
    # fig, ax = plt.subplots()
    # # for i in range(8):
    # _UE_tra = UE_posi[0,:]
    # real_part = np.real(_UE_tra.tolist())
    # imag_part = np.imag(_UE_tra.tolist())
    # ax.scatter(real_part, imag_part)
    # plt.show()


    '''生成BS位置'''
    Macro_Posi = road_cell_struct(PARAM.nCell, PARAM.Dist)
    # Macro_Posi = cell_struction(7, 150)


    '''从文件读取阴影衰落'''
    shadow_filepath = 'ShadowFad/0627_7BS_ShadowFad_dB_normed_6sigmaX_10dCov.mat'
    # shadow_filepath = 'ShadowFad/0627_7BS_ShadowFad_dB_normed_6sigmaX_10dCov.mat'
    index = 'shadowFad_dB'
    shadowFad_dB = get_shadow_from_mat(shadow_filepath, index)

    '''初始化信道、服务信息'''
    shadow = ShadowMap(shadowFad_dB)

    rate_data_all = []
    rate_data = []
    offline_rate_list = []
    num_UE = 330
    nBS = 7
    nRB = 30
    # center_cell_no = 2

    rate_list_all = []
    rec_list_all = []
    itf_list_all = []
    itf_on_ICICRB_list_all = []
    itf_on_nonICICRB_list_all = []
    est_rec_list_all = []
    est_itf_list_all = []
    est_ICIC_itf_list_all = []
    for data_num in range(26):
        print('ParaSet {}:'.format(data_num))
        root_path = 'result/0726_7BS_0.05GBR_330movingUE'
        # data_num = 0

        UE_offline_dict = np.load(root_path + '/{}/UE_offline_dict.npy'.format(data_num), allow_pickle=True).tolist()
        offline_rate = count_offline_rate(UE_offline_dict)
        # print('Offline Average rate: {:.3f}%'.format(offline_rate * 100))

        rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(data_num), allow_pickle=True)
        rate_arr = rate_arr[:400,:]
        print('Total Average rate: {:.3f}'.format(np.mean(rate_arr[rate_arr != 0]/1e6) * (1-offline_rate)))


        UE_in_different_cell_list = np.load(root_path + '/{}/UE_in_different_cell_arr.npy'.format(data_num), allow_pickle=True)
        try:
            GBR_UE_list = np.load(root_path + '/{}/GBR_UE_arr.npy'.format(data_num), allow_pickle=True)
            GBR_UE_rate = rate_arr[:, GBR_UE_list[0]]
            print('GBR Average rate: {:.3f}'.format(np.mean(GBR_UE_rate / 1e6)))
            _temp_rate = np.reshape(GBR_UE_rate,(-1,))
            print('GBR offline rate: {:.3f}%'.format(len(_temp_rate[_temp_rate<2*1e6])/len(_temp_rate) * 100))
        except:
            pass


        rec_P_list = np.load(root_path + '/{}/rec_arr.npy'.format(data_num), allow_pickle=True)
        itf_P_list = np.load(root_path + '/{}/inter_arr.npy'.format(data_num), allow_pickle=True)
        cell_rec_list = [np.array([]) for _ in range(7)]
        cell_itf_list = [np.array([]) for _ in range(7)]
        cell_rate_list = [np.array([]) for _ in range(7)]
        cell_itf_on_ICICRB_list = [np.array([]) for _ in range(7)]
        cell_itf_on_nonICICRB_list = [np.array([]) for _ in range(7)]

        est_cell_rec_list = [np.array([]) for _ in range(7)]
        est_cell_itf_on_nonICICRB_list = [np.array([]) for _ in range(7)]
        est_cell_itf_on_ICICRB_list = [np.array([]) for _ in range(7)]

        # cell_itf_on_orthogonalRB_list = [np.array([]) for _ in range(7)]
        # GBR_UE_rate = np.array([])
        try:
            RB_for_edge_ratio_list = np.load(root_path + '/{}/RB_for_edge_ratio_arr.npy'.format(data_num), allow_pickle=True)
        except:
            RB_for_edge_ratio_list = np.array([0 for _ in range(1000)])
        est_RL_state_dict = np.load(root_path + '/{}/est_RL_state_dict.npy'.format(data_num), allow_pickle=True).tolist()
        est_rec_power_list = est_RL_state_dict['est_rec_power']
        est_rec_power_list = np.reshape(est_rec_power_list, (50, -1))
        est_itf_power_list = est_RL_state_dict['est_itf_power']
        est_itf_power_list = np.reshape(est_itf_power_list, (50, -1))
        est_ICIC_itf_power_list = est_RL_state_dict['est_ICIC_itf_power']
        est_ICIC_itf_power_list = np.reshape(est_ICIC_itf_power_list, (50, -1))

        for _frame in range(len(UE_in_different_cell_list)):
            _drop = _frame * 8
            _non_ICIC_nRB = int(np.ceil((1-RB_for_edge_ratio_list[_frame]) * nRB))
            if _drop >= 400:
                break
            for _cell_no in range(nBS):
                _UE_in_cell = UE_in_different_cell_list[_frame, _cell_no]

                _rec_P = np.reshape(rec_P_list[_drop,_UE_in_cell,:],(-1,))
                _rec_P = _rec_P[_rec_P!=0]
                cell_rec_list[_cell_no] = np.concatenate((cell_rec_list[_cell_no], _rec_P))

                _itf_P = np.reshape(itf_P_list[_drop, _UE_in_cell, :], (-1,))
                _itf_P = _itf_P[_itf_P != 0]
                cell_itf_list[_cell_no] = np.concatenate((cell_itf_list[_cell_no], _itf_P))

                _itf_nonICIC_P = np.reshape(itf_P_list[_drop, _UE_in_cell, :_non_ICIC_nRB], (-1,))
                _itf_nonICIC_P = _itf_nonICIC_P[_itf_nonICIC_P != 0]
                cell_itf_on_nonICICRB_list[_cell_no] = np.concatenate((cell_itf_on_nonICICRB_list[_cell_no], _itf_nonICIC_P))

                _itf_ICIC_P = np.reshape(itf_P_list[_drop, _UE_in_cell, _non_ICIC_nRB+1:], (-1,))
                _itf_ICIC_P = _itf_ICIC_P[_itf_ICIC_P != 0]
                cell_itf_on_ICICRB_list[_cell_no] = np.concatenate((cell_itf_on_ICICRB_list[_cell_no], _itf_ICIC_P))

                _rate = np.reshape(rate_arr[_drop, _UE_in_cell], (-1,))
                _rate = _rate[_rate != 0]
                cell_rate_list[_cell_no] = np.concatenate((cell_rate_list[_cell_no], _rate))


                _UE_in_cell_idx_list = [find_UE_idx(UE_posi, _frame, UE_no) for UE_no in _UE_in_cell]
                try:
                    _est_rec_P = est_rec_power_list[_frame, _UE_in_cell_idx_list]
                except:
                    print('Wrong')
                est_cell_rec_list[_cell_no] = np.concatenate((est_cell_rec_list[_cell_no], _est_rec_P))

                _est_nonICIC_itf_P = est_itf_power_list[_frame, _UE_in_cell_idx_list]
                est_cell_itf_on_nonICICRB_list[_cell_no] = np.concatenate((est_cell_itf_on_nonICICRB_list[_cell_no], _est_nonICIC_itf_P))

                _est_ICIC_itf_P = est_ICIC_itf_power_list[_frame, _UE_in_cell_idx_list]
                est_cell_itf_on_ICICRB_list[_cell_no] = np.concatenate((est_cell_itf_on_ICICRB_list[_cell_no], _est_ICIC_itf_P))



        rate_list_all.append(cell_rate_list)
        rec_list_all.append(cell_rec_list)
        itf_list_all.append(cell_itf_list)
        itf_on_ICICRB_list_all.append(cell_itf_on_ICICRB_list)
        itf_on_nonICICRB_list_all.append(cell_itf_on_nonICICRB_list)
        est_rec_list_all.append(est_cell_rec_list)
        est_itf_list_all.append(est_cell_itf_on_nonICICRB_list)
        est_ICIC_itf_list_all.append(est_cell_itf_on_ICICRB_list)

        print('Center Cell rate: {:.3f}'.format(np.mean(rate_list_all[data_num][2]/1e6)*(1-offline_rate)))

            # _center_UE_idx = center_cell_UE_list[_frame]
            # _GBR_UE_idx = GBR_UE_list[_frame]

            # center_cell_rate = np.concatenate((center_cell_rate, np.reshape(rate_arr[:,np.array(_UE_idx)],(-1,))))
            # center_cell_rate.append(rate_arr[_drop, _center_UE_idx])
            # GBR_UE_rate.append(rate_arr[_drop,_GBR_UE_idx])
            # GBR_UE_rate = np.concatenate((GBR_UE_rate, np.reshape(rate_arr[_drop, _GBR_UE_idx], (-1,))))

        # print('Center Cell Average rate: {}'.format(np.mean(center_cell_rate[center_cell_rate != 0])))
        # print('GBR UE Average rate: {}'.format(np.mean(GBR_UE_rate[GBR_UE_rate != 0])))
        # print('Unsatified GBR UE rate : {}'.format(np.count_nonzero([GBR_UE_rate < 1e6]) / len(GBR_UE_rate)))

    # label_list = ['noICIC RB=3', 'ICIC dynamicRB', 'ICIC RB=3','ICIC RB=6','ICIC RB=9']
    # for _paraset in range(len(rate_list_all)):
    #     _rate = np.array([])
    #     for _BS_no in range(nBS):
    #         _rate = np.concatenate((_rate, np.array(rate_list_all[_paraset][_BS_no])))
    #     sns.kdeplot(_rate/1e6, label=label_list[_paraset])
    #     # sns.ecdfplot(_rate / 1e6, label=label_list[_paraset])
    # plt.xlim((0, 8))
    # plt.xlabel('bit rate(Mbps)')
    # plt.legend()
    # plt.show()

    # for _paraset in range(len(rate_list_all)):
    #     for _BS_no in range(nBS):
    #         sns.kdeplot(10*np.log10(itf_on_ICICRB_list_all[_paraset][_BS_no]), label='Cell {}'.format(_BS_no+1))
    #     plt.xlim((-100, -70))
    #     plt.xlabel('Interference(dB)')
    #     plt.legend()
    #     plt.show()





    sns.kdeplot(10 * np.log10(itf_on_ICICRB_list_all[0][2]), label='real itf power')
    sns.kdeplot(10 * np.log10(est_ICIC_itf_list_all[0][2]), label='estimated')
    # plt.xlim((-100, -60))
    plt.xlabel('Interference(dB)')
    plt.legend()
    plt.show()
