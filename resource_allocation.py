'''
本模块包含资源分配函数:
    equal_RB_allocate

暂时未考虑对不同RB的功率分配
'''

import numpy as np
from info_management import *


def equal_RB_allocate(UE_list, BS:BS, RB_per_UE, serving_map:ServingMap):
    for _UE in UE_list:
        RB_arr = BS.resource_map.RB_sorted_idx[:RB_per_UE]
        Nt_arr = np.array([])
        for _RB in RB_arr:
            if BS.resource_map.RB_ocp_num[_RB] >= BS.MaxUE_per_RB:
                return False  # BS已达到满载
            _random_Nt_range = BS.resource_map.RB_idle[_RB]
            _Nt = np.random.choice(_random_Nt_range)
            Nt_arr = np.append(Nt_arr, _Nt)

        BS.serve_UE(_UE, RB_arr, Nt_arr, serving_map)
    return True  # 成功


