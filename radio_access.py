'''
本模块包含接入方法:
    access_init
'''

from info_management import *
from resource_allocation import equal_RB_allocate
import numpy as np


def access_init(PARAMS, BS_list, UE_list, large_fading: LargeScaleFadingMap,
                serving_map: ServingMap, allocate_method=equal_RB_allocate):
    '''
    接入初始化
    :param PARAMS: 仿真参数
    :param BS_list: BS列表，元素是BS类
    :param UE_list: UE列表，元素是UE类
    :param large_fading: 大尺度衰落，LargeScaleFadingMap类
    :param serving_map: ServingMap
    :param allocate_method : 分配方法，默认为equal_RB_allocate
    :return:
    '''
    BS_no_list = []
    for _BS in BS_list:
        BS_no_list.append(_BS.no)  # 获得BS序号

    large_h = large_fading.map[BS_no_list]  # BS对应的大尺度信道
    nBS = len(BS_no_list)  # 可被接入的BS总数
    nUE = len(UE_list)

    # 根据接收最大功率准则接入，可能多个用户接入一个基站
    # BS_UE_access_list = [[] for _ in range(nBS)]
    for _UE in UE_list:
        if not _UE.active: continue
        if _UE.state != 'unserved': continue
        _UE_no = _UE.no
        # Offset = np.ones((1, nBS)) * (PARAMS.Macro.PtmaxdBm - PARAMS.Micro.PtmaxdBm - PARAMS.Micro.ABS)
        # RecivePowerdBm = large_h[:, _UE_no] - Offset
        _h = large_h[:, _UE_no]  # 所有基站到该用户的大尺度信道
        _idx = np.argsort(_h.flatten())  # 信道响应由小到大
        NewBS_idx = _idx[-1]  # 接入等效接收功率最大的基站
        # while len(serving_map.query_by_BS(BS_list[NewBS_idx].no)) == BS_list[NewBS_idx].maxUE_per_RB:
        while BS_list[NewBS_idx].if_full_load():
            # 判断BS是否达到满载，若达到则接入下一个
            _idx = _idx[:-1]
            if len(_idx) == 0: return False  # no more BS to access

            NewBS_idx = _idx[-1]

        # BS_UE_access_list[NewBS_idx].append(_UE)

        # RB_per_UE = np.floor(PARAMS.Macro.MaxUE_per_RB * PARAMS.nRB / nUE)  # 设定每个UE分配多少RB
        if allocate_method == equal_RB_allocate:
            RB_per_UE = PARAMS.RB_per_UE
            allocate_method([_UE], BS_list[NewBS_idx], int(RB_per_UE), serving_map)
        else:
            raise Exception("Invalid allocate method!", allocate_method)

    return True


if __name__ == '__main__':
    '''
    用于测试
    '''
    from simulator import Parameter
    from network_deployment import cellStructPPP
    from user_mobility import get_UE_posi_from_mat
    from channel_fading import *

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
    for i in range(PARAM.Macro.nBS):
        Macro_BS_list.append(BS(i, 'Macro', PARAM.Macro.nNt, PARAM.nRB, Macro_Posi[i], True, PARAM.Macro.MaxUE_per_RB))

    UE_list = []
    for i in range(PARAM.nUE):
        UE_list.append(UE(i, UE_posi[0,i], True))

    shadow = ShadowMap(shadowFad_dB[0])
    large_fading = LargeScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE)
    serving_map = ServingMap(PARAM.Macro.nBS, PARAM.nUE)

    _large_h = large_scale_fading(PARAM, Macro_BS_list, UE_posi[0, :], shadow, large_fading)
    print(_large_h[2, 4:6], large_fading.map[2, 4:6])  # 看更新后一不一致

    small_h = small_scale_fading(PARAM.nUE, len(Macro_BS_list), PARAM.Macro.nNt)
    print('small_h shape:', small_h.shape)

    result = access_init(PARAM, Macro_BS_list, UE_list, large_fading, serving_map)
    print(result)

