'''
本模块包含各种类，用于管理、更新各种信息:
    class UE
    class BS
    class ServingMap
    class ShadowMap
    class LargeScaleFadingMap
    class SmallScaleFadingMap
'''

import numpy as np


def delete_target_from_arr(arr, target):
    '''
    用于删除1维数组中的1个指定元素
    '''
    _idx = np.where(arr == target)[0][0]  # get the value from array in turple
    arr = np.append(arr[:_idx], arr[_idx + 1:])
    return arr

class ShadowMap:
    '''
    ShadowMap有关地理位置
    '''

    def __init__(self, shadow_data):
        self.map = shadow_data  # 单位是dB


class LargeScaleFadingMap:
    '''
    大尺度衰落=路径衰落+阴影衰落
    '''

    def __init__(self, nBS, nUE):
        self.map = np.zeros((nBS, nUE))  # 单位不是dB

    def update(self, new_fading):
        self.map = new_fading


class SmallScaleFadingMap:
    '''
    目前仅考虑平衰落，即对所有RB的小尺度衰落相同
    '''

    def __init__(self, nBS, nUE, nNt):
        self.map = np.zeros((nBS, nUE, nNt))  # 单位不是dB

    def update(self, new_fading):
        self.map = new_fading


class InstantChannelMap:
    '''
    瞬时信道
    '''

    def __init__(self, nBS, nUE, nNt):
        self.map = np.zeros((nNt, nBS, nUE))  # 单位不是dB

    def calculate_by_fading(self, large_h:LargeScaleFadingMap, small_h:SmallScaleFadingMap):
        self.map = large_h.map * np.rollaxis(small_h.map, 2)

    def update(self, new_channel):
        self.map = new_channel

class HO_state:
    def __init__(self):
        self.target_BS = -1  # 目标BS编号
        self.duration = -1  # 持续时间
        self.target_h_avg = None  # 目标BS持续时间内的平均大尺度信道
        self.h_before = None  # HO之前的服务信道
        self.failure_count = 0  # HO失败次数
        self.success_count = 0  # HO成功次数

    def update_target_BS(self, target_BS):
        self.target_BS = target_BS

    def update_duration(self, duration):
        self.duration = duration

    def update_target_h_avg(self, h_avg):
        self.target_h_avg = h_avg

    def update_h_before(self, h):
        self.h_before = h

    def add_failure_count(self):
        self.failure_count = self.failure_count + 1

    def add_success_count(self):
        self.success_count = self.success_count + 1

    def reset(self):
        self.target_BS = -1
        self.duration = -1
        self.target_h_avg = None

class UE:
    def __init__(self, no, posi, active: bool):
        self.no = no
        self.posi = posi
        self.active = active
        self.Rreq = 0
        self.state = 'unserved'  # 'served' , 'unserved', or 'handovering'
        self.state_list = ['served', 'unserved', 'handovering']
        self.serv_BS = -1
        self.RB_Nt_ocp = []  # 占用的RB_Nt,列表内的元素是元组（RB，Nt）
        self.HO_state = HO_state()

    def quit_handover(self, HO_result:bool, new_state):
        if self.state == 'handovering':
            self.update_state(new_state)
            # self.serv_BS = -1
            # self.RB_Nt_ocp = []
            self.HO_state.reset()
            if HO_result == False:
                self.HO_state.add_failure_count()  # 记录一次HO失败
            elif HO_result == True:
                self.HO_state.add_success_count()  # 记录一次HO成功

    def update_posi(self, new_posi):
        self.posi = new_posi

    def update_active(self, new_active: bool):
        self.active = new_active

    def update_Rreq(self, new_Rreq):
        self.Rreq = new_Rreq

    def update_state(self, new_state):
        if not np.isin(new_state, self.state_list):
            raise Exception("Invalid UE state!", new_state)
        self.state = new_state

    def update_serv_BS(self, new_BS):
        self.serv_BS = new_BS

    def update_RB_Nt_ocp(self, RB_Nt_list):
        self.RB_Nt_ocp = RB_Nt_list



class ResourceMap:
    def __init__(self, nRB, nNt):
        self.map = np.zeros((nRB, nNt)) - 1
        self.RB_ocp = [np.array([]) for _ in range(nRB)]  # 记录各个RB在哪些天线上服务
        self.RB_idle = [np.array(range(nNt)) for _ in range(nRB)]  # 记录各个RB在哪些天线上空闲
        self.RB_ocp_num = np.zeros((nRB,))  # 记录各个RB在多少天线上服务
        self.RB_sorted_idx = np.array(range(nRB))  # 由少到多排列占用最少的RB，以序号表示对应的RB

    def update_map(self, new_resourse_map):
        self.map = new_resourse_map

    def add_new_UE(self, UE: UE, RB_arr, Nt_arr):
        """
        给定UE、RB号和天线号，向ResourceMap添加UE服务
        :param UE: 1个UE对象
        :param RB_arr: RB号数组
        :param Nt_arr: Nt号数组
        :return: True（成功）
        """
        if len(RB_arr) != len(Nt_arr):
            raise Exception("len(RB_arr) != len(Nt_arr)", RB_arr, Nt_arr)

        RB_Nt_list = []  # 记录占用的RB_Nt,列表内的元素是数组
        for i in range(len(RB_arr)):
            _RB = int(RB_arr[i])
            _Nt = int(Nt_arr[i])
            if self.map[_RB, _Nt] != -1:
                raise Exception("Target RB has been occupied!", _RB, _Nt)

            # 更新ResourceMap
            self.map[_RB, _Nt] = UE.no
            self.RB_ocp[_RB] = np.append(self.RB_ocp[_RB], _Nt)
            self.RB_idle[_RB] = delete_target_from_arr(self.RB_idle[_RB], _Nt)
            self.RB_ocp_num[_RB] = self.RB_ocp_num[_RB] + 1
            RB_Nt_list.append((_RB, _Nt))

        # 更新RB_sorted_idx
        self.RB_sorted_idx = np.argsort(self.RB_ocp_num)

        # 改变UE对象状态
        if UE.state != 'served':
            UE.update_state('served')
        UE.update_RB_Nt_ocp(RB_Nt_list)

        return True

    def remove_UE(self, UE):
        if len(UE.RB_Nt_ocp) == 0:
            return True
        for RB_Nt in UE.RB_Nt_ocp:
            _RB, _Nt = RB_Nt
            self.map[RB_Nt] = -1
            self.RB_ocp[_RB] = delete_target_from_arr(self.RB_ocp[_RB], _Nt)
            self.RB_idle[_RB] = np.append(self.RB_idle[_RB], _Nt)
            self.RB_ocp_num[_RB] = self.RB_ocp_num[_RB] - 1

        # 改变UE对象状态
        if UE.state != 'unserved':
            UE.update_state('unserved')
        UE.update_RB_Nt_ocp([])

        return True


class ServingMap:
    def __init__(self, nBS, nUE):
        self.map = np.zeros((nBS, nUE))

    def update(self, new_map):
        self.map = new_map

    def change_by_entry(self, BS_no, UE_no, state):
        self.map[BS_no, UE_no] = state

    def query_by_UE(self, UE_no):
        col = self.map[:, UE_no]
        return np.where(col > 0)

    def query_by_BS(self, BS_no):
        row = self.map[BS_no, :]
        return np.where(row > 0)


class PrecodingInfo():
    def __init__(self):
        '''
        matrix: 干扰消除矩阵
        coeffient: 功率系数
        最终的预编码阵 = sqrt(coeffient) * matrix
        '''
        self.matrix = np.array([])
        self.coeffient = np.array([])

    def update(self, new_matrix, new_coe):
        self.matrix = new_matrix
        self.coeffient = new_coe


class BS:
    def __init__(self, no, type: str, nNt, nRB, Ptmax, posi, active: bool, MaxUE_per_RB):
        self.no = no
        self.type = type
        self.nNt = nNt
        self.nRB = nRB
        self.Ptmax = Ptmax
        self.posi = posi
        self.MaxUE_per_RB = MaxUE_per_RB
        self.active = active
        self.resource_map = ResourceMap(nRB, nNt)
        self.precoding_info = [PrecodingInfo() for _ in range(nRB)]

    def update_active(self, new_active: bool):
        self.active = new_active

    def update_precoding_matrix(self, channel: InstantChannelMap, precoding_method):
        _H = channel.map[:, self.no, :]
        for _RB in range(self.nRB):
            _RB_resource = self.resource_map.map[_RB, :]
            _serv_UE = _RB_resource[np.where(_RB_resource != -1)].astype('int32')
            if len(_serv_UE) == 0: continue  # 若没有服务用户，跳过
            '''这里的功率分配简单的以RB上的用户数作为系数'''
            _Pt_ratio = self.resource_map.RB_ocp_num[_RB] / np.sum(self.resource_map.RB_ocp_num)  # 占用的功率比例
            _W, _coe = precoding_method(_H[:, _serv_UE].T, self.Ptmax * _Pt_ratio)
            self.precoding_info[_RB].update(_W, _coe)


    def if_full_load(self):
        most_idle_RB = self.resource_map.RB_sorted_idx[0]
        if self.resource_map.RB_ocp_num[most_idle_RB] == self.MaxUE_per_RB:
            return True
        return False

    def serve_UE(self, UE: UE, RB_arr, Nt_arr, serving_map: ServingMap):
        """
        给定UE、RB号和天线号，向ResourceMap添加UE服务
        :param UE: 1个UE对象
        :param RB_arr: RB编号
        :param Nt_arr: Nt编号
        :param serving_map: BS-UE服务映射表
        :return: True（成功）
        """
        self.resource_map.add_new_UE(UE, RB_arr, Nt_arr)

        # 更新UE状态
        UE.serv_BS = self.no

        # 更新ServingMap
        serving_map.change_by_entry(self.no, UE.no, 1)
        return True

    def unserve_UE(self, UE: UE, serving_map: ServingMap):
        self.resource_map.remove_UE(UE)

        # 更新UE状态
        UE.serv_BS = -1

        # 更新ServingMap
        serving_map.change_by_entry(self.no, UE.no, 0)
        return True



