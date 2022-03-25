'''
本模块包含生成基站结构的方法:
    cellStructPPP
'''


import numpy as np


def cellStructPPP(nCell, Dist, nMicro):
    '''生成宏基站、微基站位置'''
    # nCell = PARAM.nCell  # 宏基站 / 宏小区的个数
    # Dist = PARAM.Dist  # 宏基站站间距
    Cell_r = Dist * 2 * np.tan(np.pi / 6)  # 半径

    # nMicro = np.random.poisson(PARAM.Micro.nBS_avg)  # 基站的个数，瞬时个数，每次Drop不同

    Micro_Posi = np.zeros((nCell, nMicro), dtype=np.complex_)  # 微基站位置

    if nCell == 1:
        Macro_Posi = np.array([0], dtype=np.complex_)
    elif nCell == 2:
        Macro_Posi = np.array([0, Dist], dtype=np.complex_)
    elif nCell == 3:
        Macro_Posi = np.array([0, Dist, 2 * Dist], dtype=np.complex_)
    elif nCell == 7:
        Macro_Posi = np.concatenate(([0], Dist * np.exp(1j * np.pi * (np.arange(1, 12, 2) / 6))))
    elif nCell == 19:
        Macro_Posi = np.concatenate(([0], Dist * np.exp(1j * np.pi * (np.arange(1, 12, 2) / 6)),
                                     3 ** 0.5 * Dist * np.exp(1j * np.pi * (np.arange(0, 11, 2) / 6)),
                                     2 * Dist * np.exp(1j * np.pi * (np.arange(1, 12, 2) / 6))))
    else:
        raise Exception("nCell not supported", nCell)

    # Drop micro cell
    for iCell in range(nCell):
        # generate the postion of BSs
        r_pos = Cell_r * np.sqrt(np.random.rand(100 * nMicro))
        t_pos = 2 * np.pi * np.random.rand(100 * nMicro)
        Micro_Posi[iCell, 0: nMicro] = r_pos[0: nMicro] * np.cos(t_pos[0: nMicro]) \
                                       + 1j * r_pos[0: nMicro] * np.sin(t_pos[0: nMicro]) \
                                       + Macro_Posi[iCell]

    return Macro_Posi, Micro_Posi, nMicro


if __name__ == '__main__':
    '''
    用于测试
    '''
    from simulator import Parameter
    from visualization import plot_BS_location

    PARAM = Parameter()
    Macro_Posi, Micro_Posi, nMicro = cellStructPPP(PARAM.nCell,PARAM.Dist,PARAM.Micro.nBS_avg)
    plot_BS_location(Macro_Posi, Micro_Posi)