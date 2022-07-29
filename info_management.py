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
# import channel_fading

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


class LargeScaleChannelMap:
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

    def calculate_by_fading(self, large_h:LargeScaleChannelMap, small_h:SmallScaleFadingMap):
        self.map = large_h.map * np.rollaxis(small_h.map, 2)

    def update(self, new_channel):
        self.map = new_channel

class HO_state:
    def __init__(self):
        self.handovering = False
        self.stage = None  # 'TTT' or 'HO_prep' or 'HO_exec'

        self.failure_type = 4  # 考虑的HOF有多少类型
        self.target_BS = -1  # 目标BS编号
        self.duration = -1  # 持续时间
        self.target_h = None  # 目标BS的测量信道
        self.h_before = None  # HO之前的服务信道
        self.HOF_flag = 0  # 本次HO是否已经记录过HOF

        self.success_count = 0  # HO成功次数
        self.success_posi = []  # HO成功的位置
        self.failure_count = 0  # HO失败次数
        self.failure_type_count = [0 for _ in range(self.failure_type)]  # HOF各类型次数
        self.failure_posi = [[] for _ in range(self.failure_type)]  # HO失败的位置

    def update_target_BS(self, target_BS):
        self.target_BS = target_BS

    def update_duration(self, duration):
        self.duration = duration

    def update_target_h(self, h):
        self.target_h = h

    def update_h_before(self, h):
        self.h_before = h

    def add_failure_count(self, HOF_type, posi):
        self.failure_count = self.failure_count + 1
        self.failure_type_count[HOF_type] = self.failure_type_count[HOF_type] + 1
        self.failure_posi[HOF_type].append(posi)

    def add_success_count(self, posi):
        self.success_count = self.success_count + 1
        self.success_posi.append(posi)

    def reset(self):
        self.target_BS = -1
        self.duration = -1
        self.target_h = None
        self.HOF_flag = 0


class RL_state:
    def __init__(self, active=True, Qin=-18, Qout=-20, max_period=100):
        self.active = active
        self.duration = -1
        self.Qin = Qin
        self.Qout = Qout
        self.SINR_record = []  # 用于判定RLF
        self.SINR_dB_record_all = []  # 记录仿真中所有的SINR_dB，用于仿真分析
        self.filtered_SINR_dB = None
        self.estimated_ICIC_SINR_dB = None
        self.estimated_rec_power = None
        self.estimated_itf_power = None
        self.estimated_ICIC_itf_power = None
        self.pred_SINR_dB = []  # 后几帧的预测值
        self.max_period = max_period
        self.state = 'in'

    def update_by_SINR(self, SINR, L1_filter_length):
        if not self.active:
            raise Exception("UE's RL state is not active!", self.active)

        self.SINR_dB_record_all.append(10*np.log10(SINR))
        self.SINR_record.append(SINR)
        self.SINR_record = self.SINR_record[-L1_filter_length:]
        self.filtered_SINR_dB = 10*np.log10(np.mean(self.SINR_record))

        if self.filtered_SINR_dB > self.Qin:
            self.update_state('in')
            self.reset_duration()
        elif self.filtered_SINR_dB < self.Qout:
            self.update_state('out')
            self.add_duration()
        else:
            if self.state == 'out':
                self.add_duration()

        if self.duration >= self.max_period:
            self.update_state('RLF')

        return self.state

    def update_active(self, new_active):
        self.active = new_active

    def update_state(self, new_state):
        self.state = new_state

    def add_duration(self):
        self.duration = self.duration + 1

    def reset_duration(self):
        self.duration = -1

    def reset_SINR(self):
        self.SINR_record = []
        self.filtered_SINR_dB = None




class UE:
    def __init__(self, no, type_no, posi, type, active:bool, record_len=10, GBR_flag=False, min_rate=2*1e6):
        self.no = no  # UE编号
        self.record_len = record_len
        self.posi = posi
        self.posi_record = [posi for _ in range(record_len)]
        self.future_posi = [posi for _ in range(record_len)]
        self.type = type
        self.type_no = type_no  # 对应类型中的UE编号

        self.GBR_flag = GBR_flag
        self.min_rate = min_rate

        self.active = active
        self.Rreq = 0
        self.state = 'unserved'  # 'served', 'unserved'
        # self.state_list = ['served', 'unserved']
        self.serv_BS = -1
        self.serv_BS_L3_h = None  # 服务基站的信道功率L3测量值

        self.ToS = -1  # 在当前服务小区的停留时间
        self.MTS = 100  # 最小停留时间参数
        self.RB_Nt_ocp = []  # 占用的RB_Nt,列表内的元素是元组（RB，Nt）
        self.HO_state = HO_state()


        self.neighbour_BS = []  # 邻基站列表,信道由好到差排序
        self.neighbour_BS_L3_h = []  # 邻基站的信道功率L3测量值,由好到差
        self.all_BS_L3_h_record = []
        # self.serv_BS_future_large_h = []
        # self.target_BS_future_large_h = []

        self.RL_state = RL_state()
        self.posi_type = None
        self.RB_type = None

    def estimate_needed_nRB_by_SINR(self, RB_width=180*1e3):
        try:
            if self.posi_type == 'edge':
                SINR = 10 ** (self.RL_state.estimated_ICIC_SINR_dB / 10)
            else:
                SINR = 10 ** (self.RL_state.filtered_SINR_dB / 10)

        except:
            raise Exception('RL_state.filtered_SINR_dB is None!')
        rate_per_RB = RB_width * np.log2(1 + SINR)
        needed_nRB = self.min_rate / rate_per_RB
        return np.ceil(needed_nRB)


    def is_in_invalid_RB(self, _BS, ICIC_flag:bool):
        if ICIC_flag:
            for _RB_Nt in self.RB_Nt_ocp:
                _RB = _RB_Nt[0]
                if not _RB in _BS.center_RB_idx and not _RB in _BS.edge_RB_idx:
                    return True
        else:
            for _RB_Nt in self.RB_Nt_ocp:
                _RB = _RB_Nt[0]
                if not _RB in _BS.resourse_map.RB_sorted_idx:
                    return True
        return False


    def update_posi_type(self, SINR_th, noise):
        if self.RL_state.filtered_SINR_dB != None:
            self.update_posi_type_by_SINR(SINR_th)
        else:
            e_receive_power = np.square(self.neighbour_BS_L3_h[0])
            e_interf_power = np.sum(np.square(self.neighbour_BS_L3_h[1:]))
            e_SINR = e_receive_power / (e_interf_power + noise)
            e_SINR_dB = 10 * np.log10(e_SINR)
            if e_SINR_dB > SINR_th:
                self.posi_type = 'center'
            else:
                self.posi_type = 'edge'

    def update_future_posi(self, future_posi_arr):
        self.future_posi = future_posi_arr

    def cal_future_large_h(self, PARAM, BS, shadow_map):
        import channel_fading
        future_large_fading_dB = []
        for _future_posi in self.future_posi:
            if _future_posi == None:
                _future_large_fading_dB = np.Inf
            else:
                _future_large_fading_dB = channel_fading.get_large_fading_dB_from_posi(PARAM, _future_posi, BS.posi, BS.no, shadow_map, BS.type, PARAM.scene)

            future_large_fading_dB.append(_future_large_fading_dB)

        future_large_h = 10 ** (-np.array(future_large_fading_dB) / 20)
        return future_large_h

    def quit_handover(self, HO_result, new_state, HOF_type = None):
        # if self.state == 'handovering':
        if self.HO_state.handovering:
            self.update_state(new_state)
            # self.serv_BS = -1
            # self.RB_Nt_ocp = []
            if HO_result == False:
                if HOF_type != None and self.HO_state.HOF_flag == 0:
                    self.record_HOF(HOF_type)  # 记录一次HO失败
            elif HO_result == True and self.HO_state.HOF_flag == 0:
                self.record_HOS()  # 记录一次HO成功

            self.HO_state.reset()
            self.HO_state.handovering = False
            # if new_state != 'handovering':
            #     self.HO_state.reset()

    def record_HOF(self, HOF_type):
        self.HO_state.add_failure_count(HOF_type, self.posi)

    def record_HOS(self):
        self.HO_state.add_success_count(self.posi)

    def update_RL_state_by_SINR(self, SINR, mean_filter_length):
        return self.RL_state.update_by_SINR(SINR, mean_filter_length)


    def reset_duration_and_SINR(self):
        self.RL_state.reset_duration()
        self.RL_state.reset_SINR()
        # self.RL_state.update_active(False)
        '''其余状态由BS管理'''
        # self.update_serv_BS(-1)
        # self.update_state('unserved')

    # def HO_happen(self):
    #     self.RL_state.reset_duration()
    #     self.RL_state.reset_SINR()


    def update_posi(self, new_posi):
        self.posi = new_posi
        if None in self.posi_record:
            self.posi_record = [new_posi for _ in range(self.record_len)]
        else:
            self.posi_record.append(new_posi)
            self.posi_record = self.posi_record[1:]

        if new_posi == None:
            self.update_active(False)
        else:
            self.update_active(True)

    def update_active(self, new_active: bool):
        self.active = new_active

    def update_Rreq(self, new_Rreq):
        self.Rreq = new_Rreq

    def update_state(self, new_state):
        # if not np.isin(new_state, self.state_list):
        #     raise Exception("Invalid UE state!", new_state)
        self.state = new_state

    def update_serv_BS(self, new_BS):
        self.serv_BS = new_BS

    def update_serv_BS_L3_h(self, instant_h_mean, k=0.5):
        if self.serv_BS_L3_h == None:
            self.serv_BS_L3_h = instant_h_mean
        else:
            self.serv_BS_L3_h = (1-k)*self.serv_BS_L3_h + k*instant_h_mean

    def update_all_BS_L3_h_record(self, instant_h_mean, k=0.5):
        record_len = self.record_len
        if len(self.all_BS_L3_h_record) != record_len:
            self.all_BS_L3_h_record = np.kron(instant_h_mean, np.ones((record_len, 1)))  # (15,9)
        else:
            _new_L3 = (1-k)*self.all_BS_L3_h_record[-1, :]+k*instant_h_mean  # (,9)
            self.all_BS_L3_h_record = np.concatenate((self.all_BS_L3_h_record, _new_L3[np.newaxis, :]), axis=0)  # (16,9)
            self.all_BS_L3_h_record = self.all_BS_L3_h_record[1:, :]

    def update_RB_Nt_ocp(self, RB_Nt_list):
        self.RB_Nt_ocp = RB_Nt_list

    def update_neighbour_BS(self, neighbour_list):
        self.neighbour_BS = neighbour_list

    def update_neighbour_BS_L3_h(self, BS_L3_h):
        self.neighbour_BS_L3_h = BS_L3_h

    def add_ToS(self):
        self.ToS = self.ToS + 1

    def reset_ToS(self):
        self.ToS = -1

    def update_posi_type_by_SINR(self, SINR_th):
        if self.RL_state.filtered_SINR_dB == None: return
        if self.RL_state.filtered_SINR_dB > SINR_th:
            self.posi_type = 'center'
        else:
            self.posi_type = 'edge'



class ResourceMap:
    def __init__(self, nRB, nNt, center_RB_idx, edge_RB_idx):
        self.map = np.zeros((nRB, nNt)) - 1  # 记录各个资源块上服务的UE序号
        self.RB_ocp = [np.array([]) for _ in range(nRB)]  # 记录各个RB在哪些天线上服务
        self.RB_idle_antenna = [np.array(range(nNt)) for _ in range(nRB)]  # 记录各个RB在哪些天线上空闲
        self.RB_ocp_num = np.zeros((nRB,))  # 记录各个RB在多少天线上服务
        self.RB_sorted_idx = np.array(range(nRB))  # 由少到多排列占用最少的RB，以序号表示对应的RB
        self.serv_UE_list = np.array([])
        self.extra_edge_RB_serv_list = np.array([])
        self.extra_center_RB_serv_list = np.array([])

        self.center_RB_sorted_idx = center_RB_idx
        self.edge_RB_sorted_idx = edge_RB_idx

    def add_new_extra_UE_to_list(self, UE_no, RB_type):
        if RB_type == 'center':
            self.extra_center_RB_serv_list = np.append(self.extra_center_RB_serv_list, UE_no)
        elif RB_type == 'edge':
            self.extra_edge_RB_serv_list = np.append(self.extra_edge_RB_serv_list, UE_no)


    def delete_extra_UE_from_list(self, UE_no):
        self.extra_center_RB_serv_list = self.extra_center_RB_serv_list[self.extra_center_RB_serv_list!=UE_no]
        self.extra_edge_RB_serv_list = self.extra_edge_RB_serv_list[self.extra_edge_RB_serv_list!=UE_no]

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
            self.RB_idle_antenna[_RB] = delete_target_from_arr(self.RB_idle_antenna[_RB], _Nt)
            self.RB_ocp_num[_RB] = self.RB_ocp_num[_RB] + 1
            RB_Nt_list.append((_RB, _Nt))

        # 更新RB_sorted_idx
        self.RB_sorted_idx = np.argsort(self.RB_ocp_num).astype(int)
        if len(self.center_RB_sorted_idx) != 0:
            self.center_RB_sorted_idx = self.center_RB_sorted_idx[np.argsort(self.RB_ocp_num[self.center_RB_sorted_idx.astype(int)])].astype(int)
        if len(self.edge_RB_sorted_idx) != 0:
            self.edge_RB_sorted_idx = self.edge_RB_sorted_idx[np.argsort(self.RB_ocp_num[self.edge_RB_sorted_idx.astype(int)])].astype(int)

        # 改变UE对象状态
        if UE.state == 'unserved':
            UE.update_state('served')
            '''RL state由UE管理'''
            # UE.RL_state.update_active(True)
        UE.update_RB_Nt_ocp(RB_Nt_list)
        if UE.no in self.serv_UE_list:
            raise Exception('Have same UE!')
        self.serv_UE_list = np.append(self.serv_UE_list, UE.no)

        return True

    def remove_UE(self, UE):
        if len(UE.RB_Nt_ocp) == 0:
            return True
        for RB_Nt in UE.RB_Nt_ocp:
            _RB, _Nt = RB_Nt
            self.map[RB_Nt] = -1
            self.RB_ocp[_RB] = delete_target_from_arr(self.RB_ocp[_RB], _Nt)
            self.RB_idle_antenna[_RB] = np.append(self.RB_idle_antenna[_RB], _Nt)
            self.RB_ocp_num[_RB] = self.RB_ocp_num[_RB] - 1

        # 更新RB_sorted_idx
        self.RB_sorted_idx = np.argsort(self.RB_ocp_num).astype(int)
        if len(self.center_RB_sorted_idx) != 0:
            self.center_RB_sorted_idx = self.center_RB_sorted_idx[
                np.argsort(self.RB_ocp_num[self.center_RB_sorted_idx.astype(int)])].astype(int)
        if len(self.edge_RB_sorted_idx) != 0:
            self.edge_RB_sorted_idx = self.edge_RB_sorted_idx[
                np.argsort(self.RB_ocp_num[self.edge_RB_sorted_idx.astype(int)])].astype(int)

        # 改变UE对象状态
        if UE.state == 'served':
            UE.update_state('unserved')
            '''RL state由UE管理'''
            # UE.RL_state.update_active(False)
        UE.update_RB_Nt_ocp([])
        self.serv_UE_list = self.serv_UE_list[self.serv_UE_list != UE.no]
        self.delete_extra_UE_from_list(UE.no)

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
    def __init__(self, no, type: str, nNt, nRB, Ptmax, posi, active: bool, RB_per_UE, opt_UE_per_RB, MaxUE_per_RB, center_RB_idx, edge_RB_idx, ICIC_gruop):
        self.no = no
        self.type = type
        self.nNt = nNt
        self.nRB = nRB
        self.Ptmax = Ptmax
        self.posi = posi
        self.RB_per_UE = RB_per_UE
        self.opt_UE_per_RB = opt_UE_per_RB
        self.MaxUE_per_RB = MaxUE_per_RB
        self.active = active
        self.resource_map = ResourceMap(nRB, nNt, center_RB_idx, edge_RB_idx)
        self.max_edge_UE_num = 0  # 小区的最大边缘UE数
        # self.edge_UE_num = 0  # 小区范围内的边缘UE数（包括未服务的）
        self.nUE_in_range = 0  # 小区范围内的UE数（包括未服务的）
        self.UE_in_range = []
        self.edge_UE_in_range = []  # 小区范围内的边缘UE（包括未服务的）,以SINR由小到大排序
        self.center_UE_in_range = []  # 小区范围内的中心UE（包括未服务的）,以SINR由小到大排序
        self.center_RB_idx = center_RB_idx
        self.edge_RB_idx = edge_RB_idx
        self.precoding_info = [PrecodingInfo() for _ in range(nRB)]

        self.serv_UE_list_record = []
        self.RB_ocp_num_record = []

        self.ICIC_group = ICIC_gruop

    def estimated_sum_rate(self, UE_list, PARAM, ICIC_nRB=-1, nRB_per_UE=-1):
        RB_width = PARAM.MLB.RB
        if ICIC_nRB == -1:
            ICIC_nRB = len(self.edge_RB_idx) * PARAM.ICIC.RB_partition_num
        if nRB_per_UE== -1:
            nRB_per_UE = self.RB_per_UE

        opt_ICIC_resourse = ICIC_nRB/PARAM.ICIC.RB_partition_num * self.opt_UE_per_RB
        max_ICIC_resourse = ICIC_nRB/PARAM.ICIC.RB_partition_num * self.MaxUE_per_RB
        opt_nonICIC_resourse = (self.nRB - ICIC_nRB) * self.opt_UE_per_RB
        max_nonICIC_resourse = (self.nRB - ICIC_nRB) * self.MaxUE_per_RB
        used_ICIC_resourse = 0
        used_nonICIC_resourse = 0

        UE_nonICIC_SINR_record = []
        UE_ICIC_SINR_record = []
        UE_nonICIC_nRB_record = []
        UE_ICIC_nRB_record = []
        # if self.no == 0 and ICIC_nRB==30:
        #     probe = self.no
        for _UE_no in self.UE_in_range:
            if max_ICIC_resourse+max_nonICIC_resourse-used_ICIC_resourse-used_nonICIC_resourse<nRB_per_UE:
                break

            from data_factory import search_object_form_list_by_no
            _UE = search_object_form_list_by_no(UE_list, _UE_no)
            if _UE.active == -1:
                continue
            # if _UE.serv_BS != self.no:
            #     continue

            if used_ICIC_resourse >= max_ICIC_resourse:
                _est_nonICIC_nRB = nRB_per_UE
                _est_ICIC_nRB = 0
            elif used_nonICIC_resourse >= max_nonICIC_resourse:
                _est_ICIC_nRB = nRB_per_UE
                _est_nonICIC_nRB = 0
            elif used_ICIC_resourse >= opt_ICIC_resourse and used_nonICIC_resourse < opt_nonICIC_resourse:
                if nRB_per_UE <= self.nRB-ICIC_nRB:
                    _est_nonICIC_nRB = nRB_per_UE
                    _est_ICIC_nRB = 0
                else:
                    _est_nonICIC_nRB = self.nRB-ICIC_nRB
                    _est_ICIC_nRB = nRB_per_UE-_est_nonICIC_nRB
            else:
                if nRB_per_UE <= ICIC_nRB/PARAM.ICIC.RB_partition_num:
                    _est_ICIC_nRB = nRB_per_UE
                    _est_nonICIC_nRB = 0
                else:
                    _est_ICIC_nRB = ICIC_nRB/PARAM.ICIC.RB_partition_num
                    _est_nonICIC_nRB = nRB_per_UE-_est_ICIC_nRB

            used_ICIC_resourse = used_ICIC_resourse + _est_ICIC_nRB
            used_nonICIC_resourse = used_nonICIC_resourse + _est_nonICIC_nRB



            _nonICIC_SINR = 10**(_UE.RL_state.filtered_SINR_dB/10)
            _ICIC_SINR = 10**(_UE.RL_state.estimated_ICIC_SINR_dB/10)
            UE_nonICIC_SINR_record.append(_nonICIC_SINR)
            UE_ICIC_SINR_record.append(_ICIC_SINR)
            UE_nonICIC_nRB_record.append(_est_nonICIC_nRB)
            UE_ICIC_nRB_record.append(_est_ICIC_nRB)


        K = np.min([self.opt_UE_per_RB, self.nUE_in_range])
        AG = (self.nNt - K + 1) / K
        try:
            average_UE_per_nonICIC_RB = np.floor(used_nonICIC_resourse / (self.nRB - ICIC_nRB))
            rec_on_nonICICRB_compensate_coe = (self.nNt - average_UE_per_nonICIC_RB + 1) / average_UE_per_nonICIC_RB / AG
            if rec_on_nonICICRB_compensate_coe == np.inf:
                rec_on_nonICICRB_compensate_coe = 0
        except:
            rec_on_nonICICRB_compensate_coe = 0
        try:
            average_UE_per_ICIC_RB = np.floor(used_ICIC_resourse / (ICIC_nRB / PARAM.ICIC.RB_partition_num))
            rec_on_ICICRB_compensate_coe = (self.nNt - average_UE_per_ICIC_RB + 1) / average_UE_per_ICIC_RB / AG
            if rec_on_ICICRB_compensate_coe == np.inf:
                rec_on_ICICRB_compensate_coe = 0
        except:
            rec_on_ICICRB_compensate_coe = 0

        sum_rate = 0
        for i in range(len(UE_nonICIC_SINR_record)):
            _nonICIC_SINR = UE_nonICIC_SINR_record[i] * rec_on_nonICICRB_compensate_coe
            _ICIC_SINR = UE_ICIC_SINR_record[i] * rec_on_ICICRB_compensate_coe
            _est_nonICIC_nRB = UE_nonICIC_nRB_record[i]
            _est_ICIC_nRB = UE_ICIC_nRB_record[i]
            _rate = _est_ICIC_nRB * RB_width * np.log2(1+_ICIC_SINR) + _est_nonICIC_nRB * RB_width * np.log2(1+_nonICIC_SINR)
            if np.isnan(_rate):
                raise Exception('rate is nan!')
            sum_rate = sum_rate + _rate

        return sum_rate

    def update_active(self, new_active: bool):
        self.active = new_active

    def update_precoding_matrix(self, channel: InstantChannelMap, precoding_method):
        _H = channel.map[:, self.no, :]
        for _RB in range(self.nRB):
            _RB_resource = self.resource_map.map[_RB, :]
            _serv_UE = _RB_resource[np.where(_RB_resource != -1)].astype('int32')
            if len(_serv_UE) == 0: continue  # 若没有服务用户，跳过
            # '''这里的功率分配以RB上的用户数作为系数'''
            # _Pt_ratio = self.resource_map.RB_ocp_num[_RB] / np.sum(self.resource_map.RB_ocp_num)  # 占用的功率比例
            '''这里的功率分配简单的用RB均分'''
            _Pt_ratio = 1/self.nRB
            _W, _coe = precoding_method(_H[:, _serv_UE].T, self.Ptmax * _Pt_ratio)
            self.precoding_info[_RB].update(_W, _coe)

    def get_not_full_nRB(self, max_UE_per_RB, RB_type=None):
        if RB_type == 'center':
            _RB_arr = self.resource_map.center_RB_sorted_idx
            if len(_RB_arr) != 0:
                try:
                    _RB_ocp_num = self.resource_map.RB_ocp_num[_RB_arr.astype(int)]
                except:
                    raise Exception('Get RB_ocp_num Wrong!')
                idle_nRB = len(_RB_ocp_num[_RB_ocp_num < max_UE_per_RB])
            else:
                idle_nRB = 0
        elif RB_type == 'edge':
            _RB_arr = self.resource_map.edge_RB_sorted_idx
            if len(_RB_arr) != 0:
                _RB_ocp_num = self.resource_map.RB_ocp_num[_RB_arr.astype(int)]
                idle_nRB = len(_RB_ocp_num[_RB_ocp_num < max_UE_per_RB])
            else:
                idle_nRB = 0
        else:
            _RB_arr = self.resource_map.RB_sorted_idx
            if len(_RB_arr) != 0:
                _RB_ocp_num = self.resource_map.RB_ocp_num[_RB_arr.astype(int)]
                idle_nRB = len(_RB_ocp_num[_RB_ocp_num < max_UE_per_RB])
            else:
                idle_nRB = 0
        return idle_nRB

    def if_RB_full_load(self, needed_nRB, max_UE_per_RB, RB_type=None):
        if RB_type == 'center':
            if needed_nRB > len(self.resource_map.center_RB_sorted_idx):
                return True
            most_idle_RB_arr = self.resource_map.center_RB_sorted_idx[0:needed_nRB]
        elif RB_type == 'edge':
            if needed_nRB > len(self.resource_map.edge_RB_sorted_idx):
                return True
            most_idle_RB_arr = self.resource_map.edge_RB_sorted_idx[0:needed_nRB]
        else:
            if needed_nRB > len(self.resource_map.RB_sorted_idx):
                return True
            most_idle_RB_arr = self.resource_map.RB_sorted_idx[0:needed_nRB]

        if len(most_idle_RB_arr) == 0:
            return True
        for most_idle_RB in most_idle_RB_arr:
            if self.resource_map.RB_ocp_num[int(most_idle_RB)] >= max_UE_per_RB:
                return True
        return False

    def is_full_load(self, needed_nRB, ICIC_flag):
        if ICIC_flag:
            idle_nRB = self.get_not_full_nRB(self.MaxUE_per_RB, RB_type='edge') + self.get_not_full_nRB(self.MaxUE_per_RB, RB_type='center')
            result = idle_nRB < needed_nRB
        else:
            result = self.if_RB_full_load(needed_nRB, self.MaxUE_per_RB)

        return result


    def serve_UE(self, UE: UE, RB_arr, Nt_arr, serving_map: ServingMap):
        """
        给定UE、RB号和天线号，向ResourceMap添加UE服务
        :param UE: 1个UE对象
        :param RB_arr: RB编号
        :param Nt_arr: Nt编号
        :param serving_map: BS-UE服务映射表
        :return: True（成功）
        """
        # for _RB in RB_arr:
        #     if not _RB in self.center_RB_idx and not _RB in self.edge_RB_idx:
        #         print('Wrong')

        self.resource_map.add_new_UE(UE, RB_arr, Nt_arr)

        # 更新UE状态
        UE.update_serv_BS(self.no)

        # 更新ServingMap
        serving_map.change_by_entry(self.no, UE.no, 1)
        return True

    def unserve_UE(self, UE: UE, serving_map: ServingMap):
        self.resource_map.remove_UE(UE)

        # 更新UE状态
        UE.update_serv_BS(-1)

        # 更新ServingMap
        serving_map.change_by_entry(self.no, UE.no, 0)
        return True

    def RLF_happen(self, UE: UE, serving_map: ServingMap):
        UE.reset_duration_and_SINR()
        self.unserve_UE(UE, serving_map)


