import sys
import time
import scipy.io as scio
import numpy as np

def search_object_form_list_by_no(object_list, no):
    for _obj in object_list:
        if _obj.no == no:
            return _obj


def progress_bar(pct):
    print("\r", end="")
    print("Simulation Progress: {:.2f}%: ".format(pct), "▋" * int(pct // 2), end="")
    if pct != 100:
        sys.stdout.flush()
    else:
        print('\n')


def get_data_from_mat(filepath, index):
    data = scio.loadmat(filepath)
    data = data.get(index)  # 取出字典里的label

    return data


if __name__ == '__main__':
    from simulator import *
    from channel_fading import get_shadow_from_mat
    from user_mobility import get_UE_posi_from_mat
    from network_deployment import cellStructPPP

    '''从文件读取UE位置'''
    filepath = ['Set_UE_posi_100s_500user_v{}.mat'.format(i + 1) for i in range(3)]
    index = 'Set_UE_posi'
    UE_posi = get_UE_posi_from_mat(filepath, index)

    root_path = 'result/0414_new'
    for i in range(12):
        rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(i), allow_pickle=True)
        UE_list = np.load(root_path + '/{}/UE_list.npy'.format(i), allow_pickle=True)
        HOS = [0,0,0]
        HOF = [np.array([0,0,0,0]) for _ in range(3)]
        active_UE_num = [0,0,0]
        for _UE in UE_list:
            _idx = _UE.type
            HOS[_idx] += _UE.HO_state.success_count
            HOF[_idx] += np.array(_UE.HO_state.failure_type_count)
            if _UE.active == True:
                active_UE_num[_UE.type] += 1
        print('Paraset {}'.format(i+1))
        print('Active UE: {}'.format(active_UE_num))
        print('HOS: {}, HOS rate: {:.3f}, HOF: {}'.format(np.sum(HOS), np.sum(HOS)/(np.sum(HOS)+np.sum(HOF)), np.sum(HOF)))
        for j in range(3):
            _HOS = HOS[j]
            _HOF = np.sum(HOF[j])
            _success_rate = (_HOS) / (_HOS+_HOF)
            print('UE type: {}, HOS num: {}, _success_rate: {:.3f}, HOF: {}'.format(j+1, _HOS, _success_rate, HOF[j]))


    # label_list = ['RB_per_UE={}'.format(n) for n in RB_per_UE_list]
    # label_list = ['Para Set 1']
    # plot_cdf([rate_arr[rate_arr != 0]], 'bit rate', 'cdf', label_list)


    UE_posi = UE_posi[2, :, :]
    UE_posi = process_posi_data(UE_posi)

    '''生成BS位置'''
    Macro_Posi = road_cell_struct(9, 250)