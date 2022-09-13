import numpy as np
import scipy.io as scio
import random

class Env:
    def __init__(self):
        self.state = None

    def step(self, action, cur_state, data):
        # data = scio.loadmat('./train_150-330-330e-550-770_nRB.mat')
        # data = np.array(data['sample'])

        state = self.reset(data)

        if int(np.round(action*30) > 30) or int(np.round(action*30) < 1):
            reward = np.array([0])
        else:
            nRB = np.round(action*30)
            index0 = np.where(data[:, 0] == cur_state[0])
            index1 = np.where(data[:, 1] == cur_state[1])
            index2 = np.where(data[:, 2] == cur_state[2])
            index3 = np.where(data[:, 3] == cur_state[3])
            index4 = np.where(data[:, 4] == nRB)

            index = np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(index0, index1), index2), index3),
                                   index4)

            # if index.size == 0:
            #     if nRB == 29 or nRB == 30:
            #         index4 = np.where(data[:, 4] == 28)
            #     else:
            #         if nRB % 3 == 2:
            #             index4 = np.where(data[:, 4] == nRB - 1)
            #         else:
            #             index4 = np.where(data[:, 4] == nRB + 1)
            #
            # index = np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(index0, index1), index2), index3), index4)

            bitrate = data[index, 5]
            bitrate = np.mean(bitrate)
            reward = np.array([bitrate])


        return state, reward


    def reset(self, data):

        # data = scio.loadmat('./train_150-330-330e-550-770_nRB.mat')
        # data = np.array(data['sample'])

        n = list(range(len(data)))
        i=random.sample(n, 1)

        state = np.zeros(4)
        state[0] = data[i, 0]
        state[1] = data[i, 1]
        state[2] = data[i, 2]
        state[3] = data[i, 3]

        return state
