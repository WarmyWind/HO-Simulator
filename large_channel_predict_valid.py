from large_channel_predict_train import Mydataset, get_normalize_para
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
import matplotlib.pyplot as plt
PARAM = Parameter()
batch_size = 1000
num_epochs = 500
lr = 1e-2
shadow_filepath = 'shadowFad_dB_8sigma_200dcov.mat'

if __name__ == '__main__':
    '''
    用于测试
    '''
    net = DNN_Model_Wrapper(input_dim=5 * 9 + 5 * 2, output_dim=5 * 9, no_units=100, learn_rate=lr,
                            batch_size=batch_size)
    net.load('Model/large_h_predict/DNN_0508/DNN_0508.dat')

    train_set_path = 'Dataset/large_h_dB_with_posi_train_0508.npy'
    valid_set_path = 'Dataset/large_h_dB_with_posi_valid_0508.npy'
    train_loss_path = 'Model/large_h_predict/DNN_0508/DNN_loss_train_DNN_0508.npy'
    valid_loss_path = 'Model/large_h_predict/DNN_0508/DNN_loss_valid_DNN_0508.npy'

    normalize_para_filename = 'Model/large_h_predict/DNN_0508/normalize_para.npy'
    if not os.path.isfile(normalize_para_filename):
        if os.path.isfile(train_set_path):
            dataset = np.load(train_set_path, allow_pickle=True).tolist()
            x_large_h, x_posi_real, x_posi_imag, y_large_h = dataset['0'], dataset['1'], dataset['2'], dataset['3']

            mean1, sigma1 = get_normalize_para(np.array([x_large_h,y_large_h]))

            mean2, sigma2 = get_normalize_para(np.array(x_posi_real))

            mean3, sigma3 = get_normalize_para(np.array(x_posi_imag))
        else:
            UE_posi_filepath_list = ['posi_data/v{}_2000_train.mat'.format(i + 1) for i in range(3)]
            x_large_h, x_posi_real, x_posi_imag, y_large_h = generate_dataset(shadow_filepath, UE_posi_filepath_list)
            np.save(train_set_path, {'0':x_large_h, '1':x_posi_real, '2':x_posi_imag, '3':y_large_h})

            mean1, sigma1 = get_normalize_para(np.array([x_large_h, y_large_h]))

            mean2, sigma2 = get_normalize_para(np.array(x_posi_real))

            mean3, sigma3 = get_normalize_para(np.array(x_posi_imag))

        np.save(normalize_para_filename, {'mean1': mean1, 'sigma1': sigma1, 'mean2': mean2, 'sigma2': sigma2,
                                          'mean3':mean3, 'sigma3':sigma3})
    else:
        dataset = np.load(normalize_para_filename, allow_pickle=True).tolist()
        mean1, sigma1 = dataset['mean1'], dataset['sigma1']

        mean2, sigma2 = dataset['mean2'], dataset['sigma2']

        mean3, sigma3 = dataset['mean3'], dataset['sigma3']

    if os.path.isfile(valid_set_path):
        dataset = np.load(valid_set_path, allow_pickle=True).tolist()
        x_large_h, x_posi_real, x_posi_imag, y_large_h = dataset['0'], dataset['1'], dataset['2'], dataset['3']

        x_large_h = (x_large_h - mean1) / sigma1
        y_large_h = (y_large_h - mean1) / sigma1
        x_posi_real = (x_posi_real - mean2) / sigma2
        x_posi_imag = (x_posi_imag - mean3) / sigma3

        valid_set = Mydataset(x_large_h, x_posi_real, x_posi_imag, y_large_h)
    else:
        # UE_posi_filepath_list = ['posi_data/v{}_500_valid.mat'.format(i + 1) for i in range(3)]
        UE_posi_filepath_list = ['posi_data/v3_500_valid.mat']
        x_large_h, x_posi_real, x_posi_imag, y_large_h = generate_dataset(shadow_filepath, UE_posi_filepath_list)
        np.save(valid_set_path, {'0':x_large_h, '1':x_posi_real, '2':x_posi_imag, '3':y_large_h})

        x_large_h = (x_large_h - mean1) / sigma1
        y_large_h = (y_large_h - mean1) / sigma1
        x_posi_real = (x_posi_real - mean2) / sigma2
        x_posi_imag = (x_posi_imag - mean3) / sigma3

        valid_set = Mydataset(x_large_h, x_posi_real, x_posi_imag, y_large_h)

    prediction = []
    ground_truth = []
    count = 0
    for _data in valid_set:
        x = _data[0]
        y = _data[1]
        x = torch.tensor(x)
        _pred = np.array(net.predict(x).detach().cpu())
        _denorm_pred = (_pred * sigma1) + mean1
        _pred = _pred.reshape(y.shape)
        prediction.append(_pred)
        ground_truth.append(y)
        count += 1
        if count >= 500:
            break

    prediction = np.array(prediction)
    ground_truth = np.array(ground_truth)

    # fig, ax = plt.subplots()
    # ax.plot(prediction[:, 3, 2], label='Prediction of 4 step forward')
    # ax.plot(ground_truth[:, 3, 2], label='Ground truth of 4 step forward')
    # plt.legend()
    # plt.show()

    train_loss = np.load(train_loss_path, allow_pickle=True).tolist()
    valid_loss = np.load(valid_loss_path, allow_pickle=True).tolist()
    fig, ax = plt.subplots()
    plt.yscale('log')
    ax.plot(train_loss, label='Train Loss')
    ax.plot(valid_loss, label='valid Loss')
    plt.legend()
    plt.show()

