import numpy as np


class AHO:
    def __init__(self):
        self.ideal_pred = False
        self.obs_len = 10
        self.pred_len = 10
        self.pred_allow_ratio = 1
        self.add_noise = False
        self.noise = 0.05


class PHO:
    def __init__(self):
        self.ideal_HO = False


class ICIC:
    def __init__(self):
        self.flag = True
        self.dynamic = True
        self.allow_drop_rate = 0.05

        self.RB_for_edge_ratio = 0
        self.RB_partition_num = 3

        self.edge_divide_method = 'SINR'  # 'SINR' or 'position'
        self.SINR_th = 10
        self.SINR_th_for_stat = 10
        self.edge_area_width = 30

        # edge_RB_reuse = False
        self.ideal_RL_state = True
        self.obsolete_time = 10
        self.RL_state_pred_flag = False
        self.RL_state_pred_len = 10  # max pred len refers to predictor
        self.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len

        # self.ICIC_RB_group_for_BS = [np.mod(i, 2) for i in range(8)]
        self.ICIC_RB_group_for_BS = [1,2,0,1,2,2,1]


class Macro:
    def __init__(self):
        # self.nBS = 7  # 宏基站个数
        self.nNt = 16  # 宏基站天线个数
        self.PtmaxdBm = 46  # 宏基站最大发射功率
        self.Ptmax = 10 ** (self.PtmaxdBm / 10) / 1000


class Micro:
    def __init__(self):
        self.nBS_avg = 10  # 每个宏小区微基站个数
        self.nNt = 1  # 微基站天线个数
        self.PtmaxdBm = 30  # 微基站发射功率
        self.ABS = 0


class Parameter:  # 仿真场景参数
    def __init__(self):
        self.AHO = AHO()
        self.PHO = PHO()
        self.ICIC = ICIC()
        self.Macro = Macro()
        self.Micro = Micro()

        self.nRB = 30  # RB数
        self.ICIC.RB_partition_num = 3
        self.scene = 2  # 0 = road struction, 1 = cross road struction, 2 = cell struction
        self.nCell = 7  # 小区个数
        self.Dist = 150  # 小区间距
        self.RoadWidth = self.Dist / 2 * np.sqrt(3) - 40
        self.origin_x = -250
        self.origin_y = -200

        self.Macro.nBS = self.nCell * 1  # 宏基站个数
        # self.Macro.BS_flag = np.ones((1, self.Macro.nBS))
        self.Macro.opt_UE_per_RB = np.floor(self.Macro.nNt * 0.75)  # 宏基站每个RB的最大服务用户数
        self.Micro.opt_UE_per_RB = np.floor(self.Micro.nNt)  # 微基站每个RB的最大服务用户
        # self.RB_per_UE = int(np.floor(1 * self.Macro.nBS * self.Macro.MaxUE_per_RB * self.nRB / self.nUE))

        self.dynamic_nRB_per_UE = True
        self.RB_per_UE = 3

        self.active_HO = True

        self.sigma2dBm = -95
        self.sigma2 = 10 ** (self.sigma2dBm / 10) / 1000
        self.sigma = 1.5011e-09
        self.sigma_IC = 5.0035e-10
        self.sigma_c = self.sigma2 + self.sigma
        self.sigma_e = self.sigma2 + self.sigma_IC


        self.ntype = 3
        self.nUE = 'all'  # 用户设备总数，数或‘all’
        self.nUE_per_type = 'all'  # 每种用户的数量，列表或’all‘
        self.num_neibour_BS_of_UE = 7

        self.HOM = 1  # dB
        self.TTT = 32  # 个步进时长
        self.HO_Prep_Time = 4  # HO 准备时间
        self.HO_Exec_Time = 5

        self.L3_coe = 4  # k = (1/2)^(k/4)
        self.filter_length_for_SINR = 1

        self.time_resolution = 8
        self.posi_resolution = 0.5

        self.GBR_ratio = 0  # GBR用户比例
        self.min_rate = 2*1e6  # GBR用户的最低速率



    # class RL_state:
    #     ideal_flag = True



    class pathloss:
        class Macro:
            antGaindB = 0
            dFactordB = 37.6
            pLoss1mdB = 15.3  # 36.8
            shadowdB = 6

        class Micro:
            antGaindB = 0
            dFactordB = 36.7
            pLoss1mdB = 30.6  # 36.8
            shadowdB = 6

    class MLB:
        RB = 180 * 1e3  # 180kHz
        # CBR = 3 * 1024 * 1024  # 3Mbps;恒定比特率、固定码率，形容通信服务质量


def paraset_generator():
    PARAM_list = []
    edgeRB_ratio = [0,0.2,0.4,0.6,0.8]
    RB_per_UE = [3,4,5,6,7]

    PARAM0 = Parameter()
    PARAM0.active_HO = False  # 被动切换
    PARAM0.PHO.ideal_HO = True  # 理想被动切换，哪个基站好就接入哪个
    PARAM0.TTT = 0
    PARAM0.HOM = 0
    PARAM0.AHO.ideal_pred = False
    # PARAM0.ICIC.flag = True
    PARAM0.ICIC.RB_partition_num = 3
    PARAM0.ICIC.dynamic = True  # dynamic ICIC
    # PARAM0.ICIC.RB_for_edge_ratio = _edgeRB_ratio
    # PARAM0.ICIC.SINR_th = 10
    PARAM0.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # PARAM0.ICIC.obsolete_time = 10
    PARAM0.ICIC.RL_state_pred_flag = False
    # PARAM0.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # PARAM0.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    PARAM0.dynamic_nRB_per_UE = True  # dynamic RB
    # PARAM0.RB_per_UE = _RB_per_UE
    PARAM0.nRB = 30
    PARAM0.GBR_ratio = 0
    PARAM_list.append(PARAM0)

    for _edgeRB_ratio in edgeRB_ratio:
        for _RB_per_UE in RB_per_UE:
            PARAM0 = Parameter()
            PARAM0.active_HO = False  # 被动切换
            PARAM0.PHO.ideal_HO = True  # 理想被动切换，哪个基站好就接入哪个
            PARAM0.TTT = 0
            PARAM0.HOM = 0
            PARAM0.AHO.ideal_pred = False
            # PARAM0.ICIC.flag = True
            PARAM0.ICIC.RB_partition_num = 3
            PARAM0.ICIC.dynamic = False  # fix ICIC
            PARAM0.ICIC.RB_for_edge_ratio = _edgeRB_ratio
            # PARAM0.ICIC.SINR_th = 10
            PARAM0.ICIC.ideal_RL_state = True  # ICIC时SINR理想
            # PARAM0.ICIC.obsolete_time = 10
            PARAM0.ICIC.RL_state_pred_flag = False
            # PARAM0.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
            # PARAM0.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
            PARAM0.dynamic_nRB_per_UE = False
            PARAM0.RB_per_UE = _RB_per_UE
            PARAM0.nRB = 30
            PARAM0.GBR_ratio = 0
            PARAM_list.append(PARAM0)

    # PARAM1 = Parameter()
    # PARAM1.active_HO = False  # 被动
    # PARAM1.AHO.ideal_pred = False
    # # PARAM1.ICIC.flag = True
    # PARAM1.ICIC.RB_partition_num = 3
    # PARAM1.ICIC.dynamic = False  # ICIC
    # PARAM1.ICIC.RB_for_edge_ratio = 0
    # PARAM1.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM1.ICIC.obsolete_time = 10
    # PARAM1.ICIC.RL_state_pred_flag = False
    # # PARAM1.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM1.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM1.dynamic_nRB_per_UE = False
    # PARAM1.RB_per_UE = 6
    # PARAM1.nRB = 30
    # PARAM1.GBR_ratio = 0
    # PARAM_list.append(PARAM1)
    #
    # PARAM3 = Parameter()
    # PARAM3.active_HO = False  # 被动
    # PARAM3.AHO.ideal_pred = False
    # # PARAM3.ICIC.flag = True
    # PARAM3.ICIC.RB_partition_num = 3
    # PARAM3.ICIC.dynamic = False  # ICIC
    # PARAM3.ICIC.RB_for_edge_ratio = 0
    # PARAM3.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM3.ICIC.obsolete_time = 10
    # PARAM3.ICIC.RL_state_pred_flag = False
    # # PARAM3.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM3.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM3.dynamic_nRB_per_UE = False
    # PARAM3.RB_per_UE = 9
    # PARAM3.nRB = 30
    # PARAM3.GBR_ratio = 0
    # PARAM_list.append(PARAM3)

    # PARAM0 = Parameter()
    # PARAM0.active_HO = True  # 主动切换
    # PARAM0.AHO.ideal_pred = False
    # # PARAM0.ICIC.flag = True
    # PARAM0.ICIC.RB_partition_num = 3
    # PARAM0.ICIC.dynamic = True  # ICIC
    # PARAM0.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM0.ICIC.obsolete_time = 10
    # PARAM0.ICIC.RL_state_pred_flag = False
    # # PARAM0.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM0.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM0.dynamic_nRB_per_UE = True
    # PARAM0.RB_per_UE = 3
    # PARAM0.nRB = 30
    # PARAM0.GBR_ratio = 0
    # PARAM_list.append(PARAM0)

    # PARAM1 = Parameter()
    # PARAM1.active_HO = True  # 主动切换
    # PARAM1.AHO.ideal_pred = False
    # # PARAM1.ICIC.flag = True
    # PARAM1.ICIC.RB_partition_num = 3
    # PARAM1.ICIC.dynamic = True  # ICIC
    # PARAM1.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM1.ICIC.obsolete_time = 10
    # PARAM1.ICIC.RL_state_pred_flag = False
    # # PARAM1.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM1.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM1.dynamic_nRB_per_UE = False  # 固定RB数
    # PARAM1.RB_per_UE = 3
    # PARAM1.nRB = 30
    # PARAM1.GBR_ratio = 0
    # PARAM_list.append(PARAM1)
    #
    # PARAM2 = Parameter()
    # PARAM2.active_HO = True  # 主动切换
    # PARAM2.AHO.ideal_pred = False
    # # PARAM2.ICIC.flag = True
    # PARAM2.ICIC.RB_partition_num = 3
    # PARAM2.ICIC.dynamic = True  # ICIC
    # PARAM2.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM2.ICIC.obsolete_time = 10
    # PARAM2.ICIC.RL_state_pred_flag = False
    # # PARAM2.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM2.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM2.dynamic_nRB_per_UE = False  # 固定RB数
    # PARAM2.RB_per_UE = 6
    # PARAM2.nRB = 30
    # PARAM2.GBR_ratio = 0
    # PARAM_list.append(PARAM2)
    #
    # PARAM3 = Parameter()
    # PARAM3.active_HO = True  # 主动切换
    # PARAM3.AHO.ideal_pred = False
    # # PARAM3.ICIC.flag = True
    # PARAM3.ICIC.RB_partition_num = 3
    # PARAM3.ICIC.dynamic = True  # ICIC
    # PARAM3.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM3.ICIC.obsolete_time = 10
    # PARAM3.ICIC.RL_state_pred_flag = False
    # # PARAM3.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM3.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM3.dynamic_nRB_per_UE = False  # 固定RB数
    # PARAM3.RB_per_UE = 9
    # PARAM3.nRB = 30
    # PARAM3.GBR_ratio = 0
    # PARAM_list.append(PARAM3)
    #
    #
    # PARAM4 = Parameter()
    # PARAM4.active_HO = True  # 主动切换
    # PARAM4.AHO.ideal_pred = False
    # # PARAM4.ICIC.flag = True
    # PARAM4.ICIC.dynamic = True  # ICIC
    # PARAM4.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM4.ICIC.obsolete_time = 10
    # PARAM4.ICIC.RL_state_pred_flag = False
    # # PARAM4.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM4.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM4.dynamic_nRB_per_UE = False
    # PARAM4.RB_per_UE = 6
    # # PARAM4.nRB = 15
    # PARAM_list.append(PARAM4)
    #
    #
    # PARAM5 = Parameter()
    # PARAM5.active_HO = False  # 被动切换
    # PARAM5.AHO.ideal_pred = False
    # # PARAM5.ICIC.flag = True
    # PARAM5.ICIC.dynamic = False
    # PARAM5.ICIC.RB_for_edge_ratio = 0  # 不做ICIC
    # PARAM5.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM5.ICIC.obsolete_time = 10  # 过时10帧
    # PARAM5.ICIC.RL_state_pred_flag = False
    # # PARAM5.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM5.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM5.dynamic_nRB_per_UE = False
    # PARAM5.RB_per_UE = 9
    # # PARAM5.nRB = 15
    # PARAM_list.append(PARAM5)
    #
    #
    # PARAM6 = Parameter()
    # PARAM6.active_HO = True  # 主动切换
    # PARAM6.AHO.ideal_pred = False
    # # PARAM6.ICIC.flag = True
    # PARAM6.ICIC.dynamic = True  # ICIC
    # PARAM6.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM6.ICIC.obsolete_time = 10  # 过时10帧
    # PARAM6.ICIC.RL_state_pred_flag = False
    # # PARAM6.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM6.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM6.dynamic_nRB_per_UE = False
    # PARAM6.RB_per_UE = 9
    # # PARAM6.nRB = 15
    # PARAM_list.append(PARAM6)

    # PARAM7 = Parameter()
    # PARAM7.active_HO = False  # 被动切换
    # PARAM7.AHO.ideal_pred = False
    # # PARAM7.ICIC.flag = True
    # PARAM7.ICIC.dynamic = False
    # PARAM7.ICIC.RB_for_edge_ratio = 0  # 不做ICIC
    # PARAM7.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM7.ICIC.obsolete_time = 10  # 过时10帧
    # PARAM7.ICIC.RL_state_pred_flag = False
    # # PARAM7.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM7.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM7.dynamic_nRB_per_UE = False
    # PARAM7.RB_per_UE = 12
    # # PARAM7.nRB = 15
    # PARAM_list.append(PARAM7)
    #
    #
    # PARAM8 = Parameter()
    # PARAM8.active_HO = True  # 主动切换
    # PARAM8.AHO.ideal_pred = False
    # # PARAM8.ICIC.flag = True
    # PARAM8.ICIC.dynamic = True  # ICIC
    # PARAM8.ICIC.ideal_RL_state = True  # ICIC时SINR理想
    # # PARAM8.ICIC.obsolete_time = 10  # 过时10帧
    # PARAM8.ICIC.RL_state_pred_flag = False
    # # PARAM8.ICIC.RL_state_pred_len = 10  # max pred len refers to predictor
    # # PARAM8.ICIC.dynamic_period = 10  # 每多少帧做一次动态ICIC划分,最小为1,最大为 RL_state_pred_len
    # PARAM8.dynamic_nRB_per_UE = False
    # PARAM8.RB_per_UE = 12
    # # PARAM8.nRB = 15
    # PARAM_list.append(PARAM8)

    return PARAM_list

if __name__ == '__main__':
    '''
    用于测试
    '''
    P = Parameter()
    print(P.RB_per_UE)