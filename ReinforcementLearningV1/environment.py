import numpy as np
import scipy.io as scio
import random

class Env:
    def __init__(self):
        self.state = None

    def step(self, action, cur_state):
        data = scio.loadmat('./train_240.mat')
        data = np.array(data['train_240'])
        n = list(range(len(data)))
        state = data[random.sample(n, 1),0]

        if int(np.round(action) > 7) :
            reward = np.array([0])
        else:
            nRB = np.round(action)
            index0 = np.where(data[:, 0] == cur_state)
            index2 = np.where(data[:, 2] == nRB)
            index3 = np.intersect1d(index0, index2)
            bitrate = np.mean(data[index3, 3])

            reward = np.array([bitrate])

        return state, reward


    def reset(self):

        data = scio.loadmat('./train_240.mat')
        data = np.array(data['train_240'])

        n = list(range(len(data)))

        state = data[random.sample(n, 1),0]

        return state
