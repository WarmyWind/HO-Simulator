'''
本模块包含预编码模块:
    ZF_precoding
'''


from info_management import *


def ZF_precoding(H, Ptmax):
    '''
    多用户ZF预编码
    :param H: KxNt信道矩阵，用户数K≤Nt
    :param Ptmax: 最大发射功率
    :return:  W, cof
    '''
    W = np.linalg.pinv(H)  # HW=I
    if np.square(np.linalg.norm(W)) == 0:
        print("error: W's power = 0")
    coe = Ptmax / np.square(np.linalg.norm(W))  # 功率归一化系数，coe也为接收功率
    return W, coe



