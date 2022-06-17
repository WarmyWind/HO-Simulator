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
        self.RB_partition_num = 2

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


class Macro:
    def __init__(self):
        self.nBS = 8  # 宏基站个数
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

        self.scene = 0
        self.nCell = 8  # 小区个数
        self.Dist = 150  # 小区间距
        self.RoadWidth = self.Dist / 2 * np.sqrt(3) - 40

        self.Macro.nBS = self.nCell * 1  # 宏基站个数
        # self.Macro.BS_flag = np.ones((1, self.Macro.nBS))
        self.Macro.opt_UE_per_RB = np.floor(self.Macro.nNt * 0.75)  # 宏基站每个RB的最大服务用户数
        self.Micro.opt_UE_per_RB = np.floor(self.Micro.nNt)  # 微基站每个RB的最大服务用户
        # self.RB_per_UE = int(np.floor(1 * self.Macro.nBS * self.Macro.MaxUE_per_RB * self.nRB / self.nUE))
        self.RB_per_UE = 3

        self.active_HO = True

        self.sigma2dBm = -95
        self.sigma2 = 10 ** (self.sigma2dBm / 10) / 1000
        self.sigma = 1.5011e-09
        self.sigma_IC = 5.0035e-10
        self.sigma_c = self.sigma2 + self.sigma
        self.sigma_e = self.sigma2 + self.sigma_IC

        self.nRB = 15  # RB数

        self.ntype = 3
        self.nUE = 'all'  # 用户设备总数，数或‘all’
        self.nUE_per_type = 'all'  # 每种用户的数量，列表或’all‘
        self.num_neibour_BS_of_UE = 5

        self.HOM = 1  # dB
        self.TTT = 32  # 个步进时长
        self.HO_Prep_Time = 4  # HO 准备时间
        self.HO_Exec_Time = 5

        self.L3_coe = 4  # k = (1/2)^(k/4)
        self.filter_length_for_SINR = 1

        self.posi_resolution = 8


    # class RL_state:
    #     ideal_flag = True



    class pathloss:
        class Macro:
            antGaindB = 0
            dFactordB = 37.6
            pLoss1mdB = 15.3  # 36.8
            shadowdB = 2

        class Micro:
            antGaindB = 0
            dFactordB = 36.7
            pLoss1mdB = 30.6  # 36.8
            shadowdB = 2

    class MLB:
        RB = 180 * 1e3  # 180kHz
        # CBR = 3 * 1024 * 1024  # 3Mbps;恒定比特率、固定码率，形容通信服务质量



if __name__ == '__main__':
    '''
    用于测试
    '''
    P = Parameter()
    print(P.RB_per_UE)