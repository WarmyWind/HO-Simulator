import numpy as np
from visualization import *
from network_deployment import *

class UserEquipment:
    def __init__(self, type, posi, direct):
        if type == 0:
            self.max_v = 1.5
            self.min_v = 0.5
            self.mean_v = 1
        elif type == 1:
            self.max_v = 5
            self.min_v = 3
            self.mean_v = 4
        else:
            self.max_v = 12
            self.min_v = 8
            self.mean_v = 10
        self.v = np.random.uniform(self.min_v, self.max_v)
        self.posi = posi
        self.direction = direct
        self.node_flag = [0 for _ in range(6)]  # 记录6个节点是否处理过转向

    def if_in_node(self, road_length=200, road_width=30):
        def cal_node_pct(x, start, node_length):
            return (x-start) / node_length
        half_road_width = road_width / 2
        x = np.real(self.posi)
        y = np.imag(self.posi)

        node = -1
        node_pct = -1
        if road_length-half_road_width<x<road_length+half_road_width:
            if -half_road_width<y<half_road_width:
                node = 0
                if self.direction == 1:
                    node_pct = cal_node_pct(x, road_length-half_road_width, road_width)
                else:
                    node_pct = cal_node_pct(y, -half_road_width, road_width)

            elif road_length-half_road_width<y<road_length+half_road_width:
                node = 2
                if self.direction == 1:
                    node_pct = cal_node_pct(x, road_length-half_road_width, road_width)
                else:
                    node_pct = cal_node_pct(y, road_length-half_road_width, road_width)

            elif 2*road_length-half_road_width<y<2*road_length+half_road_width:
                node = 4
                if self.direction == 1:
                    node_pct = cal_node_pct(x, road_length-half_road_width, road_width)
                else:
                    node_pct = cal_node_pct(y, 2*road_length-half_road_width, road_width)

        elif 2*road_length-half_road_width<x<2*road_length+half_road_width:
            if -half_road_width<y<half_road_width:
                node = 1
                if self.direction == 1:
                    node_pct = cal_node_pct(x, 2*road_length-half_road_width, road_width)
                else:
                    node_pct = cal_node_pct(y, -half_road_width, road_width)

            elif road_length-half_road_width<y<road_length+half_road_width:
                node = 3
                if self.direction == 1:
                    node_pct = cal_node_pct(x, 2*road_length-half_road_width, road_width)
                else:
                    node_pct = cal_node_pct(y, road_length-half_road_width, road_width)
            elif 2*road_length-half_road_width<y<2*road_length+half_road_width:
                node = 5
                if self.direction == 1:
                    node_pct = cal_node_pct(x, 2*road_length-half_road_width, road_width)
                else:
                    node_pct = cal_node_pct(y, 2*road_length-half_road_width, road_width)
        return node, node_pct

    def turn(self, node):
        if node == 0:
            direction = np.random.choice([1,1j], p=[1/3,2/3])
        elif node == 1:
            direction = 1j
        elif node == 2:
            direction = np.random.choice([1, 1j], p=[1/2, 1/2])
        elif node == 3:
            direction = 1j
        elif node == 4:
            direction = 1
        else:
            direction = 1
        self.direction = direction
        self.node_flag[node] = 1

    def move(self, delta_t):
        node, node_pct = self.if_in_node()
        if node != -1 and self.node_flag[node] == 0:
            if np.random.choice([0,1], p=[1-node_pct,node_pct]):  # 根据走过node长度的多少，决定是否要转向
                self.turn(node)  # 随机转向

        self.posi = self.posi + self.direction * self.v * delta_t
        self.new_speed()
        if self.is_out():
            self.fresh_posi()



    def new_speed(self):
        self.v = np.random.uniform(self.min_v, self.max_v)

    def is_out(self, road_length=200, road_width=30):
        x = np.real(self.posi)
        y = np.imag(self.posi)
        if x>3*road_length + road_width/2:
            return True
        return False

    def fresh_posi(self, road_width=30):
        self.posi = np.random.uniform(-road_width/2, road_width/2) + 1j * np.random.uniform(-road_width/2, road_width/2)
        self.direction = 1
        self.node_flag = [0 for _ in range(6)]

def random_start_posi(road_length=200, road_width=30):
    half_road_width = road_width / 2
    rand_road = np.random.choice(range(1, 10))
    if rand_road == 1:
        x = np.random.uniform(0, road_length)
        y = np.random.uniform(-half_road_width, half_road_width)
        direction = 1
    elif rand_road == 2:
        x = np.random.uniform(road_length, 2 * road_length)
        y = np.random.uniform(-half_road_width, half_road_width)
        direction = 1
    elif rand_road == 3:
        x = np.random.uniform(road_length - half_road_width, road_length + half_road_width)
        y = np.random.uniform(0, road_length)
        direction = 1j
    elif rand_road == 4:
        x = np.random.uniform(2 * road_length - half_road_width, 2 * road_length + half_road_width)
        y = np.random.uniform(0, road_length)
        direction = 1j
    elif rand_road == 5:
        x = np.random.uniform(road_length, 2 * road_length)
        y = np.random.uniform(road_length - half_road_width, road_length + half_road_width)
        direction = 1
    elif rand_road == 6:
        x = np.random.uniform(road_length - half_road_width, road_length + half_road_width)
        y = np.random.uniform(road_length, 2 * road_length)
        direction = 1j
    elif rand_road == 7:
        x = np.random.uniform(2 * road_length - half_road_width, 2 * road_length + half_road_width)
        y = np.random.uniform(road_length, 2 * road_length)
        direction = 1j
    elif rand_road == 8:
        x = np.random.uniform(road_length, 2 * road_length)
        y = np.random.uniform(2 * road_length - half_road_width, 2 * road_length + half_road_width)
        direction = 1
    else:
        x = np.random.uniform(2 * road_length, 3 * road_length)
        y = np.random.uniform(2 * road_length - half_road_width, 2 * road_length + half_road_width)
        direction = 1
    return x + 1j * y, direction

if __name__ == '__main__':
    def create_UElist(num, type):
        UE_list = []
        for UE_no in range(num):
            posi, direction = random_start_posi()
            UE = UserEquipment(type, posi, direction)
            UE_list.append(UE)
        return UE_list

    def get_UE_posi(UE_list):
        posi = []
        for _UE in UE_list:
            posi.append(_UE.posi)
        return posi

    UE_list = create_UElist(num=100, type=2)
    UE_posi = np.array(get_UE_posi(UE_list))[np.newaxis,:]

    # posi, direction = random_start_posi()
    # UE = UserEquipment(2, posi, direction)
    # tra = [UE.posi]

    ndrop = 1250  # 1250 * 80ms
    for drop in range(1,ndrop):
        for _UE in UE_list:
            _UE.move(delta_t=0.08)

        _posi = np.array(get_UE_posi(UE_list))[np.newaxis,:]
        UE_posi = np.concatenate((UE_posi, _posi), axis=0)

    np.save('posi_data/0511_v2_100_valid.npy', UE_posi)
    # Macro_Posi = cross_road_struction(200)
    # plot_UE_trajectory(Macro_Posi, np.array(tra))
    # plt.show()