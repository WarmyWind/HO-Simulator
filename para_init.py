import numpy as np


class Parameter:  # 仿真场景参数
    nCell = 9  # 小区个数
    Dist = 250  # 小区间距
    sigma2dBm = -95  # 接收噪声功率
    sigma2 = 10 ** (sigma2dBm / 10) / 1000
    sigma = 1.5011e-09
    sigma_IC = 5.0035e-10
    sigma_c = sigma2 + sigma
    sigma_e = sigma2 + sigma_IC
    nRB = 50  # RB数
    nUE = 300  # 用户设备总数
    nUE_per_type = 100  # 每种用户的数量

    HOM = 3  # dB
    TTT = 32  # 个步进时长

    L3_coe = 4  # k = (1/2)^(k/4)

    num_neibour_BS_of_UE = 5

    class Macro:
        nBS = 9   # 宏基站个数
        nNt = 16  # 宏基站天线个数
        PtmaxdBm = 46  # 宏基站最大发射功率
        Ptmax = 10 ** (PtmaxdBm / 10) / 1000

    class Micro:
        nBS_avg = 10  # 每个宏小区微基站个数
        nNt = 1  # 微基站天线个数
        PtmaxdBm = 30  # 微基站发射功率
        ABS = 0

    class pathloss:
        class Macro:
            antGaindB = 0
            dFactordB = 37.6
            pLoss1mdB = 15.3  # 36.8
            shadowdB = 8

        class Micro:
            antGaindB = 0
            dFactordB = 36.7
            pLoss1mdB = 30.6  # 36.8
            shadowdB = 10

    class MLB:
        RB = 180 * 1e3  # 180kHz
        # CBR = 3 * 1024 * 1024  # 3Mbps;恒定比特率、固定码率，形容通信服务质量

    def __init__(self):
        self.Macro.nBS = self.nCell * 1  # 宏基站个数
        # self.Macro.BS_flag = np.ones((1, self.Macro.nBS))
        self.Macro.MaxUE_per_RB = np.floor(self.Macro.nNt * 0.75)  # 宏基站每个RB的最大服务用户数
        # self.RB_per_UE = int(np.floor(1 * self.Macro.nBS * self.Macro.MaxUE_per_RB * self.nRB / self.nUE))
        self.RB_per_UE = 2
        self.Micro.MaxUE_per_RB = np.floor(self.Micro.nNt)  # 微基站每个RB的最大服务用户

if __name__ == '__main__':
    '''
    用于测试
    '''
    P = Parameter()
    print(P.RB_per_UE)