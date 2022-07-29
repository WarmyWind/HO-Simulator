import numpy as np
import scipy.io as scio
import random

class Env:
    def __init__(self):
        self.state = None

    def step(self, action, cur_state):
        data = scio.loadmat('./train_1234567_Nstar4-30_zscore.mat')
        data = np.array(data['sample'])

        state = self.reset()

        if int(np.round(action) > 12) or int(np.round(action) < 1):
            reward = np.array([0])
        else:
            nRB = np.round(action)/30
            index0 = np.where(data[:, 0] == cur_state[0])
            index1 = np.where(data[:, 1] == cur_state[1])
            index2 = np.where(data[:, 2] == cur_state[2])
            index3 = np.where(data[:, 3].astype(np.float32) == nRB)

            index = np.intersect1d(np.intersect1d(np.intersect1d(index0, index1), index2), index3)
            bitrate = data[index, 4]
            bitrate = np.mean(bitrate)
            reward = np.array([bitrate])

        return state, reward


    def reset(self):

        data = scio.loadmat('./train_1234567_Nstar4-30_zscore.mat')
        data = np.array(data['sample'])

        n = list(range(len(data)))
        i=random.sample(n, 1)

        state = np.zeros(3)
        state[0] = data[i, 0]
        state[1] = data[i, 1]
        state[2] = data[i, 2]

        return state
