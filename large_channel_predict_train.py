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
    def __init__(self, x_large_h, x_posi_real, x_posi_imag, y):
        posi_real = x_posi_real[:, :, np.newaxis]
        posi_imag = x_posi_imag[:, :, np.newaxis]
        x = np.float32(np.concatenate((x_large_h, posi_real, posi_imag), axis=2))
        self.data = list(zip(x, np.float32(y)))

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx]

    def __len__(self):
        return len(self.data)

shadow_filepath = '0511new_shadowFad_dB_8sigma_100dcov.mat'
train_set_path = 'Dataset/scene1_large_h_dB_with_posi_train_0511.npy'
valid_set_path = 'Dataset/scene1_large_h_dB_with_posi_valid_0511.npy'
model_name = 'scene1_DNN_0511'
normalize_para_filename = 'Model/large_h_predict/'+model_name+'/normalize_para.npy'
UE_posi_train_filepath_list = ['posi_data/0511_v{}_500_train.npy'.format(i) for i in range(3)]
UE_posi_valid_filepath_list = ['posi_data/0511_v{}_100_valid.npy'.format(i) for i in range(3)]


batch_size = 1000
num_epochs = 300
lr = 1e-2

def get_normalize_para(data):
    mean = np.mean(data)
    sigma = np.std(data)
    return mean, sigma

if __name__ == '__main__':
    save_filepath = 'Model/large_h_predict/{}'.format(model_name)
    if not os.path.exists(save_filepath):
        os.makedirs(save_filepath)

    if os.path.isfile(train_set_path):
        dataset = np.load(train_set_path, allow_pickle=True).tolist()
        x_large_h, x_posi_real, x_posi_imag, y_large_h = dataset['0'], dataset['1'], dataset['2'], dataset['3']

    else:

        x_large_h, x_posi_real, x_posi_imag, y_large_h = generate_dataset(shadow_filepath, UE_posi_train_filepath_list)
        np.save(train_set_path, {'0':x_large_h, '1':x_posi_real, '2':x_posi_imag, '3':y_large_h})


    if not os.path.isfile(normalize_para_filename):
        mean1, sigma1 = get_normalize_para(np.array([x_large_h, y_large_h]))
        mean2, sigma2 = get_normalize_para(np.array(x_posi_real))
        mean3, sigma3 = get_normalize_para(np.array(x_posi_imag))
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
    x_posi_imag = (x_posi_imag - mean3) / sigma3
    train_set = Mydataset(x_large_h, x_posi_real, x_posi_imag, y_large_h)

    if os.path.isfile(valid_set_path):
        dataset = np.load(valid_set_path, allow_pickle=True).tolist()
        x_large_h, x_posi_real, x_posi_imag, y_large_h = dataset['0'], dataset['1'], dataset['2'], dataset['3']

    else:

        x_large_h, x_posi_real, x_posi_imag, y_large_h = generate_dataset(shadow_filepath, UE_posi_valid_filepath_list)
        np.save(valid_set_path, {'0':x_large_h, '1':x_posi_real, '2':x_posi_imag, '3':y_large_h})


    x_large_h = (x_large_h - mean1) / sigma1
    y_large_h = (y_large_h - mean1) / sigma1
    x_posi_real = (x_posi_real - mean2) / sigma2
    x_posi_imag = (x_posi_imag - mean3) / sigma3
    valid_set = Mydataset(x_large_h, x_posi_real, x_posi_imag, y_large_h)


    torch.cuda.device(0)
    torch.cuda.get_device_name(torch.cuda.current_device())



    use_cuda = torch.cuda.is_available()
    if use_cuda:
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        valloader = torch.utils.data.DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, pin_memory=True)

    else:
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
        valloader = torch.utils.data.DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, pin_memory=False)


    net = DNN_Model_Wrapper(input_dim=5*9+5*2, output_dim=5*9, no_units=100, learn_rate=lr,
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


