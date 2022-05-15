'''
    本模块用于预测大尺度信道
'''


import os
import numpy as np
from utils import *
from para_init import Parameter
from channel_fading import get_shadow_from_mat
from network_deployment import road_cell_struct
from user_mobility import *
from simulator import create_Macro_BS_list
from dataset_process import generate_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from DNN_model_utils import *
PARAM = Parameter()

class Mydataset(Dataset):
    def __init__(self, x_large_h, x_posi_real, x_posi_imag, y_large_h, y_posi_real, y_posi_imag):
        x_posi_real = x_posi_real[:, :, np.newaxis]
        x_posi_imag = x_posi_imag[:, :, np.newaxis]
        x = np.float32(np.concatenate((x_large_h, x_posi_real, x_posi_imag), axis=2))
        y_posi_real = y_posi_real[:, :, np.newaxis]
        y_posi_imag = y_posi_imag[:, :, np.newaxis]
        y = np.float32(np.concatenate((y_large_h, y_posi_real, y_posi_imag), axis=2))
        self.data = list(zip(x, y))

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class My_posi_dataset(Dataset):
    def __init__(self, x_posi_real, x_posi_imag, y_posi_real, y_posi_imag):
        x_posi_real = x_posi_real[:, :, np.newaxis]
        x_posi_imag = x_posi_imag[:, :, np.newaxis]
        x = np.float32(np.concatenate((x_posi_real, x_posi_imag), axis=2))
        y_posi_real = y_posi_real[:, :, np.newaxis]
        y_posi_imag = y_posi_imag[:, :, np.newaxis]
        y = np.float32(np.concatenate((y_posi_real, y_posi_imag), axis=2))
        self.data = list(zip(x, y))

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class My_large_h_dataset(Dataset):
    def __init__(self, x_large_h, y_large_h):
        x_large_h = np.swapaxes(x_large_h, 1, 2)
        x_large_h = np.reshape(x_large_h, (-1, x_large_h.shape[-1]))
        x = np.float32(x_large_h)
        # y_posi_real = y_posi_real[:, :, np.newaxis]
        # y_posi_imag = y_posi_imag[:, :, np.newaxis]
        y_large_h = np.swapaxes(y_large_h, 1, 2)
        y_large_h = np.reshape(y_large_h, (-1, y_large_h.shape[-1]))
        y = np.float32(y_large_h)
        self.data = list(zip(x, y))
        pass

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx]

    def __len__(self):
        return len(self.data)

shadow_filepath = '0513_scene0_shadowFad_dB_8sigma_100dcov.mat'
UE_posi_train_filepath_list = ['posi_data/0514_scene0/v{}_500_train.mat'.format(i+1) for i in range(3)]
UE_posi_valid_filepath_list = ['posi_data/0514_scene0/v{}_100_valid.mat'.format(i+1) for i in range(3)]

train_set_name = 'scene0_large_h_train_0515.npy'
valid_set_name = 'scene0_large_h_valid_0515.npy'
train_set_root_path = 'Dataset/0515_scene0'
valid_set_root_path = 'Dataset/0515_scene0'
train_set_path = train_set_root_path + '/' + train_set_name
valid_set_path = valid_set_root_path + '/' + valid_set_name

model_name = 'scene0_large_h_DNN_0515'
save_filepath = 'Model/large_h_predict/{}'.format(model_name)
normalize_para_filename = 'Model/large_h_predict/'+model_name+'/normalize_para.npy'


batch_size = 1000
num_epochs = 300
lr = 1e-2

def get_normalize_para(data):
    mean = np.mean(data)
    sigma = np.std(data)
    return mean, sigma

if __name__ == '__main__':
    if not os.path.exists(train_set_root_path):
        os.makedirs(train_set_root_path)
    if not os.path.exists(valid_set_root_path):
        os.makedirs(valid_set_root_path)
    if not os.path.exists(save_filepath):
        os.makedirs(save_filepath)

    if os.path.isfile(train_set_path):
        dataset = np.load(train_set_path, allow_pickle=True).tolist()
        x_large_h, x_posi_real, x_posi_imag = dataset['0'], dataset['1'], dataset['2']
        y_large_h, y_posi_real, y_posi_imag = dataset['3'], dataset['4'], dataset['5']

    else:

        x_large_h, x_posi_real, x_posi_imag, y_large_h, y_posi_real, y_posi_imag = generate_dataset(shadow_filepath, UE_posi_train_filepath_list, PARAM, 200)
        np.save(train_set_path, {'0':x_large_h, '1':x_posi_real, '2':x_posi_imag,
                                 '3':y_large_h, '4':y_posi_real, '5':y_posi_imag})


    if not os.path.isfile(normalize_para_filename):
        mean1, sigma1 = get_normalize_para(np.array([x_large_h, y_large_h]))
        mean2, sigma2 = get_normalize_para(np.array([x_posi_real, y_posi_real]))
        mean3, sigma3 = get_normalize_para(np.array([x_posi_imag, y_posi_imag]))
        np.save(normalize_para_filename, {'mean1': mean1, 'sigma1': sigma1, 'mean2': mean2, 'sigma2': sigma2,
                                          'mean3': mean3, 'sigma3': sigma3})

    else:
        dataset = np.load(normalize_para_filename, allow_pickle=True).tolist()
        mean1, sigma1 = dataset['mean1'], dataset['sigma1']
        mean2, sigma2 = dataset['mean2'], dataset['sigma2']
        mean3, sigma3 = dataset['mean3'], dataset['sigma3']

    x_large_h = (x_large_h - mean1) / sigma1
    y_large_h = (y_large_h - mean1) / sigma1
    x_posi_real = (x_posi_real - mean2) / sigma2
    y_posi_real = (y_posi_real - mean2) / sigma2
    x_posi_imag = (x_posi_imag - mean3) / sigma3
    y_posi_imag = (y_posi_imag - mean3) / sigma3

    train_set = My_large_h_dataset(x_large_h, y_large_h)
    print('Trainset: {}'.format(len(train_set)))

    if os.path.isfile(valid_set_path):
        dataset = np.load(valid_set_path, allow_pickle=True).tolist()
        x_large_h, x_posi_real, x_posi_imag = dataset['0'], dataset['1'], dataset['2']
        y_large_h, y_posi_real, y_posi_imag = dataset['3'], dataset['4'], dataset['5']

    else:

        x_large_h, x_posi_real, x_posi_imag, y_large_h, y_posi_real, y_posi_imag = generate_dataset(shadow_filepath, UE_posi_valid_filepath_list, PARAM, 50)
        np.save(valid_set_path, {'0':x_large_h, '1':x_posi_real, '2':x_posi_imag,
                                 '3':y_large_h, '4':x_posi_real, '5':x_posi_imag})

    x_large_h = (x_large_h - mean1) / sigma1
    y_large_h = (y_large_h - mean1) / sigma1
    x_posi_real = (x_posi_real - mean2) / sigma2
    y_posi_real = (y_posi_real - mean2) / sigma2
    x_posi_imag = (x_posi_imag - mean3) / sigma3
    y_posi_imag = (y_posi_imag - mean3) / sigma3

    valid_set = My_large_h_dataset(x_large_h, y_large_h)
    print('Validset: {}'.format(len(valid_set)))


    torch.cuda.device(0)
    torch.cuda.get_device_name(torch.cuda.current_device())



    use_cuda = torch.cuda.is_available()
    if use_cuda:
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        valloader = torch.utils.data.DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, pin_memory=True)

    else:
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
        valloader = torch.utils.data.DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, pin_memory=False)


    net = DNN_Model_Wrapper(input_dim= PARAM.AHO.obs_len, output_dim=PARAM.AHO.pred_len, no_units=100, learn_rate=lr,
                                              batch_size=batch_size)
    # net.load('DNN_PRBpredict_bestnet500.dat')




    train_loss = np.zeros(num_epochs)
    valid_loss = np.zeros(num_epochs)
    _loss = np.Inf
    best_net, best_loss = None, float('inf')
    ############################################# Train #####################################
    start_epoch = 0
    net.set_train()
    tic0 = time.time()
    for i in range(start_epoch, num_epochs):

        for data in trainloader:
            x = torch.tensor(data[0])
            y = torch.tensor(data[1])
            # x = list(map(lambda x: torch.tensor(x).cuda(), data))
            fit_loss = net.fit(x, y)
            train_loss[i] += fit_loss.cpu().data.numpy()

        print('Epoch: {}'.format(i+1))
        train_loss[i] = train_loss[i] / len(train_set)
        net.scheduler.step(train_loss[i])

        valid_period = 1
        if i % valid_period == 0 or i == num_epochs - 1:

            print("Epoch: %5d/%5d, Fit loss = %8.5f" % (i+1, num_epochs, train_loss[i]))

            preds = np.zeros((1, len(valid_set)))
            for data in valloader:
                x = torch.tensor(data[0])
                y = torch.tensor(data[1])
                # x = list(map(lambda x: torch.tensor(x).cuda(), data))
                _loss = net.test(x, y)
                valid_loss[i] += _loss.cpu().data.numpy()

            valid_loss[i] = valid_loss[i] / len(valid_set)

            print("pred mse = %8.3f" % (valid_loss[i]))

            if valid_loss[i] < best_loss:
                best_loss = valid_loss[i]
                # best_net = copy.deepcopy(net.network)
                print("Save best net")

                net.save("{}/{}.dat".format(save_filepath, model_name))

        # progress_bar(i / (num_epochs - 1) * 100)

    np.save('{}\DNN_loss_train_{}'.format(save_filepath, model_name), train_loss)
    np.save('{}\DNN_loss_valid_{}'.format(save_filepath, model_name), valid_loss)


