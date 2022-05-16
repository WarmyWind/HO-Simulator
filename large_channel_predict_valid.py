from large_channel_predict_train import My_posi_dataset, get_normalize_para
import os
import numpy as np
from utils import *
from para_init import Parameter
from channel_fading import get_shadow_from_mat
from network_deployment import road_cell_struct
from user_mobility import *

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

    shadow_filepath = '0513_scene0_shadowFad_dB_8sigma_100dcov.mat'
    UE_posi_train_filepath_list = ['posi_data/0514_scene0/v{}_500_train.mat'.format(i + 1) for i in range(3)]
    UE_posi_valid_filepath_list = ['posi_data/0514_scene0/v{}_100_valid.mat'.format(i + 1) for i in range(3)]

    train_set_name = 'scene0_noise0.05_large_h_train_0515.npy'
    valid_set_name = 'scene0_noise0.05_large_h_valid_0515.npy'
    train_set_root_path = 'Dataset/0515_scene0'
    valid_set_root_path = 'Dataset/0515_scene0'
    train_set_path = train_set_root_path + '/' + train_set_name
    valid_set_path = valid_set_root_path + '/' + valid_set_name

    model_name = 'scene0_noise0.05_large_h_DNN_0515'
    save_filepath = 'Model/large_h_predict/{}'.format(model_name)
    normalize_para_filename = 'Model/large_h_predict/' + model_name + '/normalize_para.npy'

    train_loss_path = 'Model/large_h_predict/scene0_noise0.05_large_h_DNN_0515/DNN_loss_train_scene0_noise0.05_large_h_DNN_0515.npy'
    valid_loss_path = 'Model/large_h_predict/scene0_noise0.05_large_h_DNN_0515/DNN_loss_valid_scene0_noise0.05_large_h_DNN_0515.npy'

    net = DNN_Model_Wrapper(input_dim=10, output_dim=10, no_units=100, learn_rate=lr,
                            batch_size=batch_size)
    net.load(save_filepath + '/' + model_name + '.dat')

    if os.path.isfile(train_set_path):
        dataset = np.load(train_set_path, allow_pickle=True).tolist()
        x_large_h, x_posi_real, x_posi_imag = dataset['0'], dataset['1'], dataset['2']
        y_large_h, y_posi_real, y_posi_imag = dataset['3'], dataset['4'], dataset['5']

    else:

        x_large_h, x_posi_real, x_posi_imag, y_large_h, y_posi_real, y_posi_imag = generate_dataset(shadow_filepath,
                                                                                                    UE_posi_train_filepath_list)
        np.save(train_set_path, {'0': x_large_h, '1': x_posi_real, '2': x_posi_imag,
                                 '3': y_large_h, '4': y_posi_real, '5': y_posi_imag})

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
    train_set = My_posi_dataset(x_posi_real, x_posi_imag, y_posi_real, y_posi_imag)

    if os.path.isfile(valid_set_path):
        dataset = np.load(valid_set_path, allow_pickle=True).tolist()
        x_large_h, x_posi_real, x_posi_imag = dataset['0'], dataset['1'], dataset['2']
        y_large_h, y_posi_real, y_posi_imag = dataset['3'], dataset['4'], dataset['5']

    else:

        x_large_h, x_posi_real, x_posi_imag, y_large_h, y_posi_real, y_posi_imag = generate_dataset(shadow_filepath,
                                                                                                    UE_posi_valid_filepath_list)
        np.save(valid_set_path, {'0': x_large_h, '1': x_posi_real, '2': x_posi_imag,
                                 '3': y_large_h, '4': x_posi_real, '5': x_posi_imag})

    x_large_h = (x_large_h - mean1) / sigma1
    y_large_h = (y_large_h - mean1) / sigma1
    x_posi_real = (x_posi_real - mean2) / sigma2
    y_posi_real = (y_posi_real - mean2) / sigma2
    x_posi_imag = (x_posi_imag - mean3) / sigma3
    y_posi_imag = (y_posi_imag - mean3) / sigma3
    valid_set = My_posi_dataset(x_posi_real, x_posi_imag, y_posi_real, y_posi_imag)

    # prediction = []
    # ground_truth = []
    # count = 0
    # for _data in valid_set:
    #     x = _data[0]
    #     y = _data[1]
    #     x = torch.tensor(x)
    #     _pred = np.array(net.predict(x).detach().cpu())
    #     # _denorm_pred = (_pred * sigma1) + mean1
    #     _pred = _pred.reshape(y.shape)
    #     prediction.append(_pred)
    #     ground_truth.append(y)
    #     count += 1
    #     if count >= 10000:
    #         break
    #
    # prediction = np.array(prediction)
    # ground_truth = np.array(ground_truth)
    #
    # MSE = np.mean(np.square(prediction - ground_truth), axis=2)
    # MSE = np.mean(MSE, axis=0)
    # RMSE = np.sqrt(MSE)
    # print('MSE', MSE)
    # print('RMSE', RMSE)


    # fig, ax = plt.subplots()
    # ax.plot(prediction[1500:2000, 7, 1], label='Prediction of 640ms forward')
    # ax.plot(ground_truth[1500:2000, 7, 1], label='Ground truth of 640ms forward')
    # plt.grid()
    # plt.legend()
    # plt.show()

    train_loss = np.load(train_loss_path, allow_pickle=True).tolist()
    valid_loss = np.load(valid_loss_path, allow_pickle=True).tolist()
    fig, ax = plt.subplots()
    plt.yscale('log')
    ax.plot(train_loss, label='Train Loss')
    ax.plot(valid_loss, label='valid Loss')
    print(train_loss[-1], valid_loss[-1])
    print(np.sqrt(train_loss[-1]),np.sqrt(valid_loss[-1]))
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()

    UE_posi_filepath = ['0511_v{}_500.npy'.format(i) for i in range(3)]
    # UE_posi_filepath = ['0511_v0_500.npy']
    posi_index = 'Set_UE_posi'
    UE_posi = get_UE_posi_from_file(UE_posi_filepath, posi_index)
    # UE_posi = UE_posi[2, :, :]
    UE_posi = process_posi_data(UE_posi)

    # example_car_posi = UE_posi[2][:, 0]
    # for idrop in range(len(example_car_posi)):
    #     x =

