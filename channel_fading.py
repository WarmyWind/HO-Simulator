'''
本模块包含关于衰落信道的函数:
    get_shadow_from_mat
    large_scale_fading
    small_scale_fading
'''


from info_management import *
import numpy as np
import scipy.io as scio


def get_shadow_from_mat(filepath, index):
    data = scio.loadmat(filepath)
    shadow_data = data.get(index)  # 取出字典里的label

    return shadow_data


def large_scale_fading(PARAMS, BS_list, UE_list, shadow_map:ShadowMap):
    '''
    大尺度信道衰落=路径衰落+阴影衰落
    :param PARAMS: 仿真参数
    :param BS_list: BS列表，元素是BS类
    :param UE_list: UE列表，元素是UE类
    :param shadow_map: 阴影衰落地图，与地理位置有关
    :param large_fading: 大尺度衰落，LargeScaleFadingMap类
    :return: large_scale_fading，1个二维数组表示BS-UE的大尺度衰落
    '''

    nBS = len(BS_list)  # 所有基站数
    nUE = len(UE_list)  # 所有用户数

    large_scale_fading_dB = np.zeros((nBS, nUE))
    for iUE in range(nUE):
        for iBS in range(nBS):
            if BS_list[iBS].type == 'Macro':
                antGain = PARAMS.pathloss.Macro.antGaindB
                dFactor = PARAMS.pathloss.Macro.dFactordB
                pLoss1m = PARAMS.pathloss.Macro.pLoss1mdB
                # shadow  = PARAMS.pathloss.Macro.shadowdB
            else:
                antGain = PARAMS.pathloss.Micro.antGaindB
                dFactor = PARAMS.pathloss.Micro.dFactordB
                pLoss1m = PARAMS.pathloss.Micro.pLoss1mdB
                # shadow  = PARAMS.pathloss.Micro.shadowdB


            if not UE_list[iUE].active:
                large_scale_fading_dB[iBS, iUE] = -np.Inf
            else:
                distServer = np.abs(UE_list[iUE].posi - BS_list[iBS].posi)  # 用户-基站距离
                '''
                下面的x_temp和y_temp后加的常数与数据有关
                '''
                x_temp = int(np.ceil(np.real(UE_list[iUE].posi)/0.5))
                y_temp = int(np.ceil((np.imag(UE_list[iUE].posi)-PARAMS.Dist/2/np.sqrt(3))/0.5))
                x_temp = np.min((shadow_map.map.shape[2]-1, x_temp))
                y_temp = np.min((shadow_map.map.shape[1]-1, y_temp))
                shadow = shadow_map.map[iBS][y_temp, x_temp]
                large_scale_fading_dB[iBS, iUE] = pLoss1m + dFactor * np.log10(distServer) + shadow - antGain

    large_scale_fading = 10 ** (-large_scale_fading_dB / 20)
    # print('大尺度衰落(dB)：',large_scale_fading_dB[:,0])
    # large_fading.update(large_scale_fading)
    return large_scale_fading


def small_scale_fading(nUE, nBS, nNt, fading_model='Rayleigh'):
    small_H = np.zeros((nBS,nUE, nNt),dtype=np.complex_)

    if fading_model == 'Rayleigh':
        for iBS in range(nBS):

            small_h = np.zeros((nUE, nNt),dtype=np.complex_)
            for iUE in range(nUE):
                # 控制随机数相同
                np.random.seed(iBS * nUE + iUE)
                h = (np.random.randn(nNt) + 1j * np.random.randn(nNt)) / np.sqrt(2)
                small_h[iUE,:] = h  # 小尺度衰落

            small_H[iBS] = small_h
    else:
        raise Exception("fading_model not supported", fading_model)

    return small_H


if __name__ == '__main__':
    '''
    用于测试
    '''
    from simulator import Parameter
    from network_deployment import cellStructPPP
    from user_mobility import get_UE_posi_from_mat

    PARAM = Parameter()
    filepath = 'shadowFad_dB1.mat'
    index = 'shadowFad_dB'
    shadowFad_dB = get_shadow_from_mat(filepath, index)
    print('shadowFad_dB shape:',shadowFad_dB.shape)
    filepath = 'Set_UE_posi_60s_250user_1to2_new.mat'
    index = 'Set_UE_posi'
    UE_posi = get_UE_posi_from_mat(filepath, index)

    Macro_Posi, Micro_Posi, nMicro = cellStructPPP(PARAM.nCell,PARAM.Dist, PARAM.Micro.nBS_avg)
    Macro_BS_list = []

    for i in range(PARAM.Macro.nBS):
        Macro_BS_list.append(BS(i, 'Macro', PARAM.Macro.nNt, PARAM.nRB,PARAM.Macro.Ptmax, Macro_Posi[i], True, PARAM.Macro.MaxUE_per_RB))

    shadow = ShadowMap(shadowFad_dB[0])
    print(shadow.map.shape)  # (3,)
    large_fading = LargeScaleFadingMap(PARAM.Macro.nBS, PARAM.nUE)

    large_h = large_scale_fading(PARAM, Macro_BS_list, UE_posi[0,:], shadow)  # (3, 250)
    large_fading.update(large_h)
    print('大尺度信道shape：',large_fading.map.shape)
    print('大尺度信道：', large_fading.map[:,0])
    print('大尺度衰落（dB）：', -20 * np.log10(large_fading.map[:,0]))


    small_h = small_scale_fading(PARAM.nUE, len(Macro_BS_list), PARAM.Macro.nNt)
    print(small_h.shape)
