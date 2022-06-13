'''
本模块包含预编码模块:
    ZF_precoding
'''


from info_management import *


def ZF_precoding(H, Ptmax, alloc = 'alloc_fair'):
    '''
    多用户ZF预编码
    :param H: KxNt信道矩阵，用户数K≤Nt
    :param Ptmax: 最大发射功率
    :return:  W, coe
    '''
    W = np.linalg.pinv(H)  # HW=I
    if np.square(np.linalg.norm(W)) == 0:
        print("error: W's power = 0")

    if alloc == 'rec_fair':  # 接收功率相等
        coe = Ptmax / np.square(np.linalg.norm(W))  # 功率归一化系数，coe也为接收功率
        coe = np.ones(H.shape[0],) * coe
    elif alloc == 'alloc_fair':  # 分配功率相等
        coe = Ptmax / H.shape[0] / np.square(np.linalg.norm(W, axis=0))
    else:
        raise Exception("Invalid ZF allocate method!", alloc)
    # precoding_power = coe * np.square(np.linalg.norm(W, axis=0))
    # precoding_power_sum = np.sum(precoding_power)
    return W, coe



