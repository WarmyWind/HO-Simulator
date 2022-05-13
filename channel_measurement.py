'''
本模块包含信道测量相关函数:
    update_target_BS_L3_h
'''

from info_management import *
import numpy as np


def update_all_BS_L3_h_record(UE_list, large_channel: LargeScaleChannelMap, instant_channel: InstantChannelMap, L3_coe, ideal_meassure=True):
    k = (1 / 2) ** (L3_coe / 4)
    instant_h = instant_channel.map
    for _UE in UE_list:
        if not _UE.active: continue  # 如果UE不活动，则跳过
        # if _UE.state == 'unserved':  continue  # 如果UE不被服务，则跳过
        if ideal_meassure:
            _large_h = large_channel.map[:, _UE.no]
            _UE.update_all_BS_L3_h_record(_large_h)
        else:
            _instant_h = instant_h[:, :, _UE.no]
            _instant_h_power = np.square(np.abs(_instant_h))
            _instant_h_power_mean = np.mean(_instant_h_power, axis=0)  # (9,)
            _UE.update_all_BS_L3_h_record(np.sqrt(_instant_h_power_mean), k)


def update_serv_BS_L3_h(UE_list, large_channel: LargeScaleChannelMap, instant_channel: InstantChannelMap, L3_coe, ideal_meassure=True):
    k = (1 / 2) ** (L3_coe / 4)
    instant_h = instant_channel.map
    for _UE in UE_list:
        if not _UE.active: continue  # 如果UE不活动，则跳过
        if _UE.state == 'unserved':  continue  # 如果UE不被服务，则跳过
        if ideal_meassure:
            _large_h = large_channel.map[_UE.serv_BS, _UE.no]
            _UE.update_serv_BS_L3_h(_large_h)
        else:
            _instant_h = instant_h[:, _UE.serv_BS, _UE.no]
            _instant_h_power = np.square(np.abs(_instant_h))
            _instant_h_power_mean = np.mean(_instant_h_power, axis=0)
            _UE.update_serv_BS_L3_h(np.sqrt(_instant_h_power_mean), k)
