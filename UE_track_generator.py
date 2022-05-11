import numpy as np


class UserEquipment:
    def __init__(self, type, posi, direct):
        if type == 1:
            self.max_v = 1.5
            self.min_v = 0.5
            self.mean_v = 1
        elif type == 2:
            self.max_v = 5
            self.min_v = 3
            self.mean_v = 4
        else:
            self.max_v = 12
            self.min_v = 8
            self.mean_v = 10
        self.posi = posi
        self.direction = direct
        self.node_flag = [0 for _ in range(6)]  # 记录6个节点是否处理过转向

    def if_in_node(self, road_length=200, road_width=30):
        half_road_width = road_width / 2
        x = np.real(self.posi)
        y = np.imag(self.posi)
        if road_length-half_road_width<x<road_length+half_road_width:
            if -half_road_width<y<half_road_width: return 0
            elif road_length-half_road_width<y<road_length+half_road_width: return 2
            elif 2*road_length-half_road_width<y<2*road_length+half_road_width: return 4
        elif 2*road_length-half_road_width<x<2*road_length+half_road_width:
            if -half_road_width<y<half_road_width: return 1
            elif road_length-half_road_width<y<road_length+half_road_width: return 3
            elif 2*road_length-half_road_width<y<2*road_length+half_road_width: return 5
        return -1



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
