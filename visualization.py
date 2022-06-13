'''
本模块包含可视化方法:
    plot_BS_location
    plot_UE_trajectory
    plot_cdf
    plot_bar
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.io as scio
from info_management import *
from network_deployment import *
import scipy.io as sio
import seaborn as sns

def plot_hexgon(ax, center, dist):
    radius = dist/np.sqrt(3)
    for _center in center:
        point_list=[]
        for angle in np.arange(np.pi/6, 2*np.pi+np.pi/6, np.pi/3):
            point_list.append(_center + radius*np.exp(1j*angle))

        for i in range(len(point_list)):
            _point1 = point_list[i]
            _point2 = point_list[(i+1) % len(point_list)]
            ax.plot(np.real([_point1, _point2]), np.imag([_point1, _point2]), color='silver')

    return ax

def plot_BS_location(Macro_Posi, Micro_Posi = None, ax=None):
    if ax == None:
        fig, ax = plt.subplots()
    ax.scatter(np.real(Macro_Posi), np.imag(Macro_Posi), label='Macro BS')
    if Micro_Posi != None:
        ax.scatter(np.real(Micro_Posi), np.imag(Micro_Posi), label='Micro BS')
    # plt.legend()
    # plt.show()
    return ax


def plot_UE_trajectory(Macro_Posi, UE_tra, label_list=None, ax=None):
    if ax == None:
        fig, ax = plt.subplots()
    ax.scatter(np.real(Macro_Posi), np.imag(Macro_Posi), label='Macro BS')
    dist = np.abs(Macro_Posi[0]-Macro_Posi[1])
    ax = plot_hexgon(ax, Macro_Posi, dist)
    if len(UE_tra.shape) == 2:
        for i in range(UE_tra.shape[-1]):
            _UE_tra = UE_tra[:, i]
            _UE_tra = _UE_tra[np.where(_UE_tra != None)]
            if label_list == None:
                ax.plot(np.real(_UE_tra.tolist()), np.imag(_UE_tra.tolist()), label='User{}'.format(i))
            else:
                ax.plot(np.real(_UE_tra.tolist()), np.imag(_UE_tra.tolist()), label=label_list[i])
    elif len(UE_tra.shape) == 1:
        UE_tra = UE_tra[np.where(UE_tra != None)]
        ax.plot(np.real(UE_tra.tolist()), np.imag(UE_tra.tolist()), label='User')

    plt.legend()
    # plt.show()
    return ax


def plot_cdf(data, xlabel, ylabel, label_list, cumulative=True, normed=1):
    # data is a list of array
    fig, ax = plt.subplots()
    for i in range(len(data)):
        _d = np.array(data[i]).flatten()
        ax.hist(_d, bins=250, density=normed, cumulative=cumulative, histtype='step', label=label_list[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_xbound(np.min(data), np.max(data))

    if cumulative:
        ax.set_ybound(0, 1)
        fix_hist_step_vertical_line_at_end(ax)
    # plt.legend(loc=loc)
    # plt.show()
    return ax


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


def plot_bar(data, xlabel, ylabel, para_list, label_list, ax=None):
    if ax == None:
        fig, ax = plt.subplots()

    # 创建分组柱状图，需要自己控制x轴坐标
    xticks = np.arange(len(para_list))


    # 注意控制柱子的宽度，这里选择0.25
    for i in range(len(label_list)):
        ax.bar(xticks + i * 0.25, data[i], width=0.25, label=label_list[i])

    if xlabel != '':
        ax.set_xlabel(xlabel)
    if ylabel != '':
        ax.set_ylabel(ylabel)
    ax.legend()


    # 最后调整x轴标签的位置
    ax.set_xticks(xticks + 0.25)
    ax.set_xticklabels(para_list)

    # plt.legend(loc=loc)
    # plt.show()
    return ax

def plot_HO_count_bar(ax, para_list, HOS, HOF, tick_label, width, xtick_bias):
    bar = []
    xticks = np.arange(len(para_list))
    # for q in range(len(para_list)):
    #     # _HOF = np.sum(HOF, axis=0)
    bar1 = ax.bar(xticks+xtick_bias, HOS, width=width, color='green', tick_label=tick_label)
    bar.append(bar1)
    bar2 = ax.bar(xticks+xtick_bias, HOF[:,1], width=width, bottom=HOS, color='red', tick_label=tick_label)
    bar.append(bar2)
    bottom = []

    bar3 = ax.bar(xticks, HOF[:,3], width=width, bottom=HOF[:,1]+HOS, color='yellow', tick_label=tick_label)
    bar.append(bar3)
    return ax, bar

def plot_rate_map(BS_posi, UE_posi, rate_data, title_list, fineness=20, loc='upper right'):
    x = np.real(UE_posi)
    y = np.imag(UE_posi)
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    interval_x = x_range / fineness
    interval_y = y_range / fineness

    # interval = np.max((interval_x,interval_y))
    # interval_x = interval
    # interval_y = interval

    def trans_to_map_index(posi):
        _x = np.real(posi)
        _idx_x = (_x - np.min(x) + interval_x / 2) // interval_x
        _y = np.imag(posi)
        _idx_y = (_y - np.min(y) + interval_y / 2) // interval_y
        return np.array([_idx_x, _idx_y]).astype('int32')

    num_plot = len(rate_data)
    plt.figure(figsize=(4 * (num_plot) + 1, 4))
    # grid = plt.GridSpec(1, 4*(num_plot) + 1, wspace=0.5, hspace=0.5)
    grid = plt.GridSpec(1, 4 * (num_plot) + 1)
    axes = []
    for i in range(num_plot):
        _ax = plt.subplot(grid[:, i * 4:(i + 1) * 4])
        axes.append(_ax)

    last_ax = plt.subplot(grid[:, -1])
    axes.append(last_ax)


    map_list = []
    for i in range(num_plot):
        map = np.zeros((fineness + 1, fineness + 1))
        map_cnt = np.zeros((fineness + 1, fineness + 1))

        if len(UE_posi.shape == 2):
            for nDrop in range(UE_posi.shape[0]):
                for _UE_no in range(UE_posi.shape[1]):
                    _idx_x, _idx_y = trans_to_map_index(UE_posi[nDrop, _UE_no])

                    map[_idx_x, _idx_y] = map[_idx_x, _idx_y] + rate_data[i][nDrop, _UE_no]
                    map_cnt[_idx_x, _idx_y] = map_cnt[_idx_x, _idx_y] + 1
        else:
            raise Exception("UE_posi shape is not supported", UE_posi.shape)

        map[map != 0] = map[map != 0] / map_cnt[map_cnt != 0]  # 平均速率
        map_list.append(map)

    map_rate_max = np.max(map_list)
    map_rate_min = np.min(map_list)
    norm = mpl.colors.Normalize(vmin=map_rate_min, vmax=map_rate_max)
    for i in range(num_plot):
        ax = axes[i]
        map = map_list[i]
        ax.imshow(map.transpose(), norm=norm, cmap='Blues', origin='lower')
        _BS_x = (np.real(BS_posi) - np.min(x)) / interval_x
        _BS_y = (np.imag(BS_posi) - np.min(y)) / interval_y
        ax.scatter(_BS_x, _BS_y, c='r', label='BS')
        ax.legend(loc=loc)
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        add_scalr_bar(ax, 17.5, 20, 0, interval_x * 2.5, fineness)
        add_scalr_bar(ax, 0, 2.5, 0, interval_y * 2.5, fineness, 'horizontal')
        ax.set_xticks(np.arange(0, fineness + 1, fineness / 8))
        ax.set_yticks(np.arange(0, fineness + 1, fineness / 8))
        ax.set_title(title_list[i])

    # 展示colorbar
    ax = axes[-1]
    # norm = mpl.colors.Normalize(vmin=map_rate_min, vmax=map_rate_max)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Blues'),
                 cax=ax, orientation='vertical', label='Bit Rate')
    # 避免图片显示不完全
    plt.tight_layout()
    # plt.show()


def add_scalr_bar(ax, x_start, x_end, y, length, fineness, orientation='vertical'):
    if orientation == 'vertical':
        ax.hlines(y=y, xmin=x_start, xmax=x_end, colors='black', ls='-', lw=1, label='{} km'.format(length))
        ax.vlines(x=x_start, ymin=y - fineness / 80, ymax=y + fineness / 80, colors='black', ls='-', lw=1)
        ax.vlines(x=x_end, ymin=y - fineness / 80, ymax=y + fineness / 80, colors='black', ls='-', lw=1)
        ax.text((x_start + x_end) / 2, y + fineness / 40, '{:.1f}m'.format(length), horizontalalignment='center')
    elif orientation == 'horizontal':
        y_start = x_start
        y_end = x_end
        x = y
        ax.vlines(x=x, ymin=y_start, ymax=y_end, colors='black', ls='-', lw=1, label='{} km'.format(length))
        ax.hlines(y=y_start, xmin=x - fineness / 80, xmax=x + fineness / 80, colors='black', ls='-', lw=1)
        ax.hlines(y=y_end, xmin=x - fineness / 80, xmax=x + fineness / 80, colors='black', ls='-', lw=1)
        ax.text(x + fineness / 80, (y_start + y_end) / 2, '{:.1f}m'.format(length), verticalalignment='center',
                rotation=270)

def plot_HO_map(UE_list, BS_posi, UE_tra, label_list=None):
    ax = plot_UE_trajectory(BS_posi, UE_tra, label_list)

    for idx in range(len(UE_list)):
        UE = UE_list[idx]
        HOF_posi = UE.HO_state.failure_posi
        HOS_posi = UE.HO_state.success_posi
        color_list = ['black', 'red', 'darkorange', 'blue']
        for i in range(len(HOF_posi)):
            # _some_type_HOF_posi = HOF_posi[i]
            _posi = HOF_posi[i]
            if idx == 0:
                ax.scatter(np.real(_posi), np.imag(_posi), marker='|', s=100, color=color_list[i], label='HOF type{}'.format(i+1))
            else:
                ax.scatter(np.real(_posi), np.imag(_posi), marker='|', s=100, color=color_list[i])


        if idx == 0:
            ax.scatter(np.real(HOS_posi), np.imag(HOS_posi), marker='d', s=20, color='darkgreen', label='HOS')
        else:
            ax.scatter(np.real(HOS_posi), np.imag(HOS_posi), marker='d', s=20, color='darkgreen')

    plt.legend()
    # plt.show()
    return ax

def plot_large_channel(PARAM, BS_posi, BS_no_list, shadow_map, UE_posi, UE_HOF_posi = None, L3=True):
    antGain = PARAM.pathloss.Macro.antGaindB
    dFactor = PARAM.pathloss.Macro.dFactordB
    pLoss1m = PARAM.pathloss.Macro.pLoss1mdB
    UE_posi = UE_posi[np.where(UE_posi != None)].tolist()
    # probe = np.real(UE_posi)
    fig, ax = plt.subplots()
    for BS_no in BS_no_list:
        distServer = np.abs(UE_posi - BS_posi[BS_no])  # 用户-基站距离
        x_temp = (np.ceil(np.real(UE_posi) / 0.5)).astype(int)
        y_temp = (np.ceil((np.imag(UE_posi) - PARAM.Dist / 2 / np.sqrt(3)) / 0.5)).astype(int)
        large_scale_fading_dB = []
        for _drop in range(len(x_temp)):
            _x_temp = np.min((shadow_map.map.shape[2] - 1, x_temp[_drop]))
            _y_temp = np.min((shadow_map.map.shape[1] - 1, y_temp[_drop]))
            shadow = shadow_map.map[BS_no][_y_temp, _x_temp]
            _large_scale_fading_dB = pLoss1m + dFactor * np.log10(distServer[_drop]) + shadow - antGain
            large_scale_fading_dB.append(_large_scale_fading_dB)

        large_scale_fading = 10**(np.array(large_scale_fading_dB)/10)
        L3_large_scale_fading = [large_scale_fading[0]]
        if L3:
            for i in range(1, len(large_scale_fading)):
                L3_large_scale_fading.append(0.5*L3_large_scale_fading[i-1] + 0.5*large_scale_fading[i])
            large_scale_fading_dB = 10*np.log10(L3_large_scale_fading)

        # x = [j*PARAM.posi_resolution for j in range(len(large_scale_fading))]
        x = np.linspace(np.min(np.real(UE_posi)),np.max(np.real(UE_posi)),len(large_scale_fading))
        ax.plot(x, -large_scale_fading_dB, label='BS{}'.format(BS_no))

    if UE_HOF_posi != None:
        color_list = ['black', 'red', 'darkorange', 'blue']
        for HOF_type in range(len(UE_HOF_posi)):
            color = color_list[HOF_type]
            for j in range(len(UE_HOF_posi[HOF_type])):
                _HOF_posi = UE_HOF_posi[HOF_type][j]
                if j == 0:
                    ax.axvline(np.real(_HOF_posi), c=color, lw=0.5, label='HOF type{}'.format(HOF_type+1))
                else:
                    ax.axvline(np.real(_HOF_posi), c=color, lw=0.5)
    # plt.xlabel('Time(ms)')
    plt.xlabel('x(m)')
    plt.xticks()
    plt.ylabel('Large Fading(dB)')
    plt.legend(loc='lower left')
    plt.show()
    return ax

def plot_SINR(UE_posi, UE, UE_HOF_posi = None, Qout = -8, ax=None):
    if ax == None:
        fig, ax = plt.subplots()

    # inactive_drop = np.where(UE_posi != None)[0][0] * PARAM.posi_resolution
    # _failure_posi = UE.HO_state.failure_posi[HOF_type][0]
    # _posi_idx = np.where(UE_posi == _failure_posi)[0][0]
    # _drop_idx = _posi_idx * PARAM.posi_resolution
    _posi_without_None = UE_posi[np.where(UE_posi != None)].tolist()
    _probe = len(UE.RL_state.SINR_dB_record_all)
    x = np.linspace(np.min(np.real(_posi_without_None)), np.max(np.real(_posi_without_None)),
                        len(UE.RL_state.SINR_dB_record_all))
    ax.plot(x, UE.RL_state.SINR_dB_record_all)

    ax.axhline(Qout, label='Qout', c='grey')

    if UE_HOF_posi != None:
        color_list = ['black', 'red', 'darkorange', 'blue']
        for HOF_type in range(len(UE_HOF_posi)):
            color = color_list[HOF_type]
            for j in range(len(UE_HOF_posi[HOF_type])):
                _HOF_posi = UE_HOF_posi[HOF_type][j]
                if j == 0:
                    ax.axvline(np.real(_HOF_posi), c=color, lw=0.5, label='HOF type{}'.format(HOF_type+1))
                else:
                    ax.axvline(np.real(_HOF_posi), c=color, lw=0.5)
    plt.xlabel('x(m)')
    plt.xticks()
    plt.ylabel('SINR(dB)')
    plt.legend(loc='lower left')
    plt.show()
    return ax

def plot_road(ax, scene, Dist, RoadWidth):
    if scene == 0:
        origin_y_point = (Dist / 2 * np.sqrt(3) - RoadWidth) / 2
        ax.axhline(origin_y_point, c='black', ls='-', lw=1)
        ax.axhline(origin_y_point+RoadWidth, c='black', label='road', ls='-', lw=1)
    else:
        ax.hlines(y=-15, xmin=-15, xmax=415, colors='black', ls='-', lw=1)
        ax.hlines(y=15, xmin=-15, xmax=185, colors='black', ls='-', lw=1)
        ax.hlines(y=15, xmin=215, xmax=385, colors='black', ls='-', lw=1)
        ax.hlines(y=185, xmin=215, xmax=385, colors='black', ls='-', lw=1)
        ax.hlines(y=215, xmin=215, xmax=385, colors='black', ls='-', lw=1)
        ax.hlines(y=385, xmin=215, xmax=385, colors='black', ls='-', lw=1)
        ax.hlines(y=385, xmin=415, xmax=615, colors='black', ls='-', lw=1)
        ax.hlines(y=415, xmin=185, xmax=615, colors='black', ls='-', lw=1)

        ax.vlines(x=185, ymin=15, ymax=415, colors='black', ls='-', lw=1)
        ax.vlines(x=215, ymin=15, ymax=185, colors='black', ls='-', lw=1)
        ax.vlines(x=215, ymin=215, ymax=385, colors='black', ls='-', lw=1)
        ax.vlines(x=385, ymin=15, ymax=185, colors='black', ls='-', lw=1)
        ax.vlines(x=385, ymin=215, ymax=385, colors='black', ls='-', lw=1)
        ax.vlines(x=415, ymin=-15, ymax=385, colors='black', ls='-', lw=1, label='road')
    return ax

if __name__ == '__main__':
    from simulator import *
    from channel_fading import get_shadow_from_mat
    from user_mobility import get_UE_posi_from_file
    from network_deployment import cellStructPPP

    PARAM = Parameter()
    '''从文件读取UE位置'''
    UE_posi_filepath = 'UE_tra/0527_scene0/Set_UE_posi_240.mat'
    index = 'Set_UE_posi'
    UE_posi = get_UE_posi_from_file(UE_posi_filepath, index)
    # UE_posi = UE_posi[2, :, :]
    if len(UE_posi.shape) != 3:
        UE_posi = np.swapaxes(UE_posi, 0,1)
        UE_posi = np.reshape(UE_posi, (PARAM.ntype, -1, UE_posi.shape[1]))
        UE_posi = np.swapaxes(UE_posi, 1, 2)
    UE_posi = process_posi_data(UE_posi)

    # root_path = 'result/0527_ideal_AHO_scene0'
    # data_num = 1
    # rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(data_num), allow_pickle=True)
    # # print('Total Average rate: {}'.format(np.mean(rate_arr[rate_arr != 0])))
    # UE_list = np.load(root_path + '/{}/UE_list.npy'.format(data_num), allow_pickle=True)
    # BS_list = np.load(root_path + '/{}/BS_list.npy'.format(data_num), allow_pickle=True)
    # label_list = ['RB_per_UE={}'.format(n) for n in RB_per_UE_list]
    # label_list = ['Para Set 1']
    # plot_cdf([rate_arr[rate_arr != 0]], 'bit rate', 'cdf', label_list)
    # plt.show()


    '''生成BS位置'''
    Macro_Posi = road_cell_struct(PARAM.nCell, PARAM.Dist)

    # ax = plot_BS_location(Macro_Posi)
    # dist = np.abs(Macro_Posi[0] - Macro_Posi[1])
    # ax = plot_hexgon(ax, Macro_Posi, dist)
    #
    # '''绘制道路'''
    # ax = plot_road(ax,PARAM.scene, PARAM.Dist, PARAM.RoadWidth)
    # plt.axis('square')
    # plt.xlim(-10, 1100)
    # plt.ylim(-110, 300)
    # plt.legend()
    # plt.show()
    #
    # example_car_tra = UE_posi[2][:, 5]
    # ax = plot_UE_trajectory(Macro_Posi, example_car_tra)
    # ax = plot_road(ax, PARAM.scene, PARAM.Dist, PARAM.RoadWidth)
    # plt.axis('square')
    # # plt.xlim(-10, 1100)
    # # plt.ylim(-210, 426.5)
    # # plt.xlim(-30, 630)
    # # plt.ylim(-30, 430)
    # plt.xlim(-10, 1100)
    # plt.ylim(-110, 300)
    # plt.legend()
    # plt.show()

    '''从文件读取阴影衰落'''
    shadow_filepath = 'ShadowFad/0523_ShadowFad_dB_normed_6sigmaX_10dCov.mat'
    index = 'shadowFad_dB'
    shadowFad_dB = get_shadow_from_mat(shadow_filepath, index)

    '''初始化信道、服务信息'''
    shadow = ShadowMap(shadowFad_dB)

    # '''绘制大尺度信道信息'''
    # BS_no_list = [i for i in range(PARAM.nCell)]
    # plot_large_channel(PARAM, Macro_Posi, BS_no_list, shadow, UE_posi[2][:,0])


    # '''绘制UE例子的HO地图'''
    # example_UE_posi=[]
    # example_UE_list = []
    # type_no = [14,13,0]  # 选取三个不同种类的UE编号
    # for i in range(3):
    #     example_UE_posi.append(UE_posi[i][:,type_no[i]])
    #     example_UE_list.append(UE_list[i*50+type_no[i]])
    #
    # example_UE_posi = np.transpose(example_UE_posi)
    # label = ['pedestrian','bike','car']
    # plot_HO_map(example_UE_list, Macro_Posi, np.array(example_UE_posi), label_list=label)
    # # fig, ax = plot_UE_trajectory(Macro_Posi, np.array(example_UE_posi), label_list=label)
    # # plt.legend()
    # plt.grid()
    # plt.axis('square')
    # plt.xlim(-10, 1100)
    # plt.ylim(-210, 426.5)
    # plt.show()


    def handle_HO_rate(observe_length, PARAM, UE_list, UE_posi):
        # observe_length = 8
        HO_duration_rate = [[] for _ in range(4+1)]
        for i in range(len(UE_list)):
            _UE = UE_list[i]
            _UE_posi = UE_posi[_UE.type][:,_UE.type_no]
            for j in range(len(_UE.HO_state.failure_posi)):
                for _failure_posi in _UE.HO_state.failure_posi[j]:
                    _posi_idx = np.where(_UE_posi == _failure_posi)[0][0]
                    _drop_idx = _posi_idx*PARAM.posi_resolution
                    _rate_arr = rate_arr[_drop_idx-observe_length:_drop_idx, i]
                    if len(_rate_arr) != 0:
                        HO_duration_rate[j].append(_rate_arr)

            for _success_posi in _UE.HO_state.success_posi:
                _posi_idx = np.where(_UE_posi == _success_posi)[0][0]
                _drop_idx = _posi_idx * PARAM.posi_resolution
                _rate_arr = rate_arr[_drop_idx - observe_length-9:_drop_idx-9, i]
                if len(_rate_arr) != 0:
                    HO_duration_rate[4].append(_rate_arr)

        HO_duration_rate.append(rate_arr[rate_arr != 0])
        HO_duration_rate_all = np.array([])
        for i in range(5):
            if len(HO_duration_rate[i]) == 0:
                continue
            else:
                if len(HO_duration_rate_all) == 0:
                    HO_duration_rate_all = np.array(HO_duration_rate[i])
                else:
                    HO_duration_rate_all = np.concatenate((HO_duration_rate_all, np.array(HO_duration_rate[i])), axis=0)

        return HO_duration_rate_all.reshape(-1)

    def plot_all_HO_posi(UE_list, UE_posi, consider_HOF, consider_HOS=True, ax=None):
        if ax == None:
            fig, ax = plt.subplots()

        HOF_posi = [[] for _ in range(4)]
        HOS_posi = []
        for i in range(len(UE_list)):
            _UE = UE_list[i]
            _UE_posi = UE_posi[_UE.type][:,_UE.type_no]
            for j in range(len(_UE.HO_state.failure_posi)):
                if j+1 in consider_HOF:
                    for _failure_posi in _UE.HO_state.failure_posi[j]:
                        HOF_posi[j].append(_failure_posi)

            if consider_HOS:
                for _success_posi in _UE.HO_state.success_posi:
                    _posi_idx = np.where(_UE_posi == _success_posi)[0][0]
                    HOS_posi.append(_success_posi)

        for i in range(4):
            if len(HOF_posi[i]) != 0:
                ax.scatter(np.real(HOF_posi[i]), np.imag(HOF_posi[i]), marker='o', s=10, label='HOF{}'.format(i+1))
        if consider_HOS:
            ax.scatter(np.real(HOS_posi), np.imag(HOS_posi), marker='d', s=10, color='darkgreen', label='HOS')
        return ax, HOF_posi


    # '''绘制BS和道路'''
    # ax = plot_BS_location(Macro_Posi)
    # dist = np.abs(Macro_Posi[0] - Macro_Posi[1])
    # ax = plot_hexgon(ax, Macro_Posi, dist)
    #
    # ax = plot_road(ax, PARAM.scene, PARAM.Dist, PARAM.RoadWidth)
    #
    # consider_HOF = [1,2,3,4]
    # ax, HOF_posi = plot_all_HO_posi(UE_list, UE_posi, consider_HOF,consider_HOS=True, ax=ax)
    # plt.axis('square')
    # plt.xlim(-10, 1100)
    # plt.ylim(-110, 400)
    # # plt.ylim(10, 110)
    # plt.legend()
    # plt.show()
    # # sio.savemat(root_path + '/{}/HOM=0_TTT=0_pingpong_posi.mat'.format(data_num),{'Set_pingpong_posi':HOF_posi[3]})
    #
    # fig, ax = plt.subplots()
    #
    # for BS_no in range(1, 6):
    #     example_BS = BS_list[BS_no]
    #     serv_UE_list = example_BS.serv_UE_list_record
    #     serv_UE_num = []
    #     for _serv_UE in serv_UE_list:
    #         serv_UE_num.append(len(_serv_UE))
    #
    #     ax.plot(serv_UE_num, label='BS{}'.format(BS_no))
    #
    # plt.legend()
    # plt.show()
    #
    # SINR_dB_record = []
    # _temp_SINR_dB = np.array([])
    # for _UE in UE_list:
    #
    #     # _temp_SINR_dB = _UE.RL_state.SINR_dB_record_all[::8]
    #     if len(_temp_SINR_dB) < 1240:
    #         if len(_temp_SINR_dB) + len(_UE.RL_state.SINR_dB_record_all[::8]) <= 1240:
    #             _temp_SINR_dB = np.append(_temp_SINR_dB, _UE.RL_state.SINR_dB_record_all[::8])
    #             if len(_temp_SINR_dB) == 1240:
    #                 SINR_dB_record.append(_temp_SINR_dB)
    #                 _temp_SINR_dB = np.array([])
    #         elif len(_temp_SINR_dB) + len(_UE.RL_state.SINR_dB_record_all[::8]) > 1240:
    #             _temp_SINR_dB = np.array(_UE.RL_state.SINR_dB_record_all[::8])
    #
    #     print('UEtype:{}  record_len:{}'.format(_UE.type, len(SINR_dB_record)))

    # sio.savemat(root_path + '/{}/76v1_55v2_43v3_HOM=3_TTT=640_SINR_dB.mat'.format(data_num), {'SINR_dB': np.array(SINR_dB_record)})

    def handle_func(max_inter_arr, UE_offline_dict):
        max_interf_edge_all = np.array([])
        max_interf_center_all = np.array([])
        # UE_on_edge_RB_num_list = []
        for i in range(len(max_inter_arr)):
            edge_UE_arr = np.array([])
            center_UE_arr = np.array([])
            _max_inter = max_inter_arr[i]
            _edge_UE_all = UE_offline_dict['edge_UE'][i]
            # UE_on_edge_RB_list = UE_offline_dict['UE_on_edge_RB'][i]

        #     _num = 0
        #     for j in range(len(UE_on_edge_RB_list)):
        #         _num = _num + len(UE_on_edge_RB_list[j])
        #     UE_on_edge_RB_num_list.append(_num)

            for _edge_UE_list in _edge_UE_all:
                edge_UE_arr = np.concatenate((edge_UE_arr, np.array(_edge_UE_list)))
            max_interf_edge_all = np.concatenate((max_interf_edge_all, _max_inter[edge_UE_arr.astype(int)]))

            _center_UE_all = UE_offline_dict['center_UE'][i]
            for _center_UE_list in _center_UE_all:
                center_UE_arr = np.concatenate((center_UE_arr, np.array(_center_UE_list)))
            max_interf_center_all = np.concatenate((max_interf_center_all, _max_inter[center_UE_arr.astype(int)]))

        return max_interf_center_all, max_interf_edge_all

    observe_length = 8
    # HO_duration_rate_all = handle_HO_rate(observe_length, PARAM, UE_list, UE_posi)
    # rate_data = [HO_duration_rate_all]
    rate_data_all = []
    rate_data = []


    root_path = 'result/0609_scene0'
    data_num = 0
    # drop_idx = 500
    rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(data_num), allow_pickle=True)
    print('Total Average rate: {}'.format(np.mean(rate_arr[rate_arr != 0])))
    rate_data_all.append(rate_arr)

    UE_list = np.load(root_path + '/{}/UE_list.npy'.format(data_num), allow_pickle=True)
    HO_duration_rate_all = handle_HO_rate(observe_length, PARAM, UE_list, UE_posi)
    rate_data.append(HO_duration_rate_all)

    UE_offline_dict = np.load(root_path + '/{}/UE_offline_dict.npy'.format(data_num), allow_pickle=True).tolist()
    max_inter_arr = np.load(root_path + '/{}/max_inter_arr.npy'.format(data_num), allow_pickle=True)
    # RB_for_edge_ratio_arr = np.load(root_path + '/{}/RB_for_edge_ratio_arr.npy'.format(data_num), allow_pickle=True)
    # plt.plot(RB_for_edge_ratio_arr)
    # plt.show()
    # edge_UE_list = UE_offline_dict['edge_UE'][drop_idx]
    # UE_on_edge_RB_list = UE_offline_dict['UE_on_edge_RB'][drop_idx]

    # fig, ax = plt.subplots()
    # ax.plot(UE_on_edge_RB_num_list, label='30m')
    # max_interf_center_all, max_interf_edge_all = handle_func(max_inter_arr, UE_offline_dict)
    # max_interf_edge_all_list = [10*np.log10(copy.deepcopy(max_interf_edge_all+PARAM.sigma2))]
    # max_interf_center_all_list = [10*np.log10(copy.deepcopy(max_interf_center_all+PARAM.sigma2))]


    root_path = 'result/0609_scene0_test_AHO'
    data_num = 0
    rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(data_num), allow_pickle=True)
    print('Total Average rate: {}'.format(np.mean(rate_arr[rate_arr != 0])))
    rate_data_all.append(rate_arr)

    UE_list = np.load(root_path + '/{}/UE_list.npy'.format(data_num), allow_pickle=True)
    HO_duration_rate_all = handle_HO_rate(observe_length, PARAM, UE_list, UE_posi)
    rate_data.append(HO_duration_rate_all)

    UE_offline_dict = np.load(root_path + '/{}/UE_offline_dict.npy'.format(data_num), allow_pickle=True).tolist()
    max_inter_arr = np.load(root_path + '/{}/max_inter_arr.npy'.format(data_num), allow_pickle=True)

    # max_interf_center_all, max_interf_edge_all = handle_func(max_inter_arr, UE_offline_dict)
    # max_interf_edge_all_list.append(10*np.log10(copy.deepcopy(max_interf_edge_all+PARAM.sigma2)))
    # max_interf_center_all_list.append(10*np.log10(copy.deepcopy(max_interf_center_all+PARAM.sigma2)))




    root_path = 'result/0609_scene0_test_AHO'
    data_num = 1
    rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(data_num), allow_pickle=True)
    print('Total Average rate: {}'.format(np.mean(rate_arr[rate_arr != 0])))
    rate_data_all.append(rate_arr)

    UE_list = np.load(root_path + '/{}/UE_list.npy'.format(data_num), allow_pickle=True)
    HO_duration_rate_all = handle_HO_rate(observe_length, PARAM, UE_list, UE_posi)
    rate_data.append(HO_duration_rate_all)
    # UE_offline_dict = np.load(root_path + '/{}/UE_offline_dict.npy'.format(data_num), allow_pickle=True).tolist()
    # max_inter_arr = np.load(root_path + '/{}/max_inter_arr.npy'.format(data_num), allow_pickle=True)

    # max_interf_center_all, max_interf_edge_all = handle_func(max_inter_arr, UE_offline_dict)
    # max_interf_edge_all_list.append(10 * np.log10(copy.deepcopy(max_interf_edge_all + PARAM.sigma2)))
    # max_interf_center_all_list.append(10 * np.log10(copy.deepcopy(max_interf_center_all + PARAM.sigma2)))

    root_path = 'result/0609_scene0'
    data_num = 1
    rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(data_num), allow_pickle=True)
    print('Total Average rate: {}'.format(np.mean(rate_arr[rate_arr != 0])))
    rate_data_all.append(rate_arr)

    UE_list = np.load(root_path + '/{}/UE_list.npy'.format(data_num), allow_pickle=True)
    HO_duration_rate_all = handle_HO_rate(observe_length, PARAM, UE_list, UE_posi)
    rate_data.append(HO_duration_rate_all)
    # UE_offline_dict = np.load(root_path + '/{}/UE_offline_dict.npy'.format(data_num), allow_pickle=True).tolist()
    # max_inter_arr = np.load(root_path + '/{}/max_inter_arr.npy'.format(data_num), allow_pickle=True)

    root_path = 'result/0609_scene0'
    data_num = 2
    rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(data_num), allow_pickle=True)
    print('Total Average rate: {}'.format(np.mean(rate_arr[rate_arr != 0])))
    rate_data_all.append(rate_arr)

    UE_list = np.load(root_path + '/{}/UE_list.npy'.format(data_num), allow_pickle=True)
    HO_duration_rate_all = handle_HO_rate(observe_length, PARAM, UE_list, UE_posi)
    rate_data.append(HO_duration_rate_all)

    root_path = 'result/0609_scene0'
    data_num = 4
    rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(data_num), allow_pickle=True)
    print('Total Average rate: {}'.format(np.mean(rate_arr[rate_arr != 0])))
    rate_data_all.append(rate_arr)

    UE_list = np.load(root_path + '/{}/UE_list.npy'.format(data_num), allow_pickle=True)
    HO_duration_rate_all = handle_HO_rate(observe_length, PARAM, UE_list, UE_posi)
    rate_data.append(HO_duration_rate_all)


    # ax.plot(UE_on_edge_RB_num_list, label='75m')
    # plt.xlim(1,10000)

    # rate_arr_no_zero = rate_data_all[0]
    # rate_arr_no_zero=rate_arr_no_zero[rate_arr_no_zero!=0]
    # sns.kdeplot(rate_data[0], label='passive')
    #
    # # rate_arr_no_zero = rate_data_all[1]
    # # rate_arr_no_zero = rate_arr_no_zero[rate_arr_no_zero != 0]
    # sns.kdeplot(rate_data[1], label='ICIC passive')
    #
    # # rate_arr_no_zero = rate_data_all[2]
    # # rate_arr_no_zero = rate_arr_no_zero[rate_arr_no_zero != 0]
    # sns.kdeplot(rate_data[2], label='ICIC ideal active')
    #
    # # rate_arr_no_zero = rate_data_all[3]
    # # rate_arr_no_zero = rate_arr_no_zero[rate_arr_no_zero != 0]
    # sns.kdeplot(rate_data[3], label='ICIC active')
    #
    # plt.xlim((-1*1e6,8*1e6))
    # plt.legend()
    # plt.show()


    # root_path = 'result/0530_ICIC_AHO_scene0'
    # data_num = 4
    # rate_arr = np.load(root_path + '/{}/rate_arr.npy'.format(data_num), allow_pickle=True)
    # print('Total Average rate: {}'.format(np.mean(rate_arr[rate_arr != 0])))
    # UE_list = np.load(root_path + '/{}/UE_list.npy'.format(data_num), allow_pickle=True)
    # HO_duration_rate_all = handle_HO_rate(observe_length, PARAM, UE_list, UE_posi)
    # rate_data.append(HO_duration_rate_all)


    # '''绘制HO前的速率cdf'''
    # label_list=['passive','ideal active', 'active', 'ICIC passive', 'ICIC ideal active', 'ICIC active']
    # # label_list = ['nRB=13', 'nRB=15', 'nRB=17']
    # ax = plot_cdf(np.array(rate_data)/1e6, 'bit rate(Mbps)', 'cdf', label_list)
    # plt.legend(loc='lower right')
    # plt.xlim(0, 6)
    # plt.show()
    # print(np.mean(rate_data[0]),np.mean(rate_data[1]),np.mean(rate_data[2]),
    #       np.mean(rate_data[3]),np.mean(rate_data[4]),np.mean(rate_data[5]))

    rate_data = np.array(rate_data_all)
    sio.savemat('passive_rate.mat', {'passive_rate': rate_data[0][rate_data[0] != 0].reshape((-1))})
    sio.savemat('ideal_active_rate.mat', {'ideal_active_rate':rate_data[1][rate_data[1] != 0].reshape((-1))})
    sio.savemat('active_rate.mat', {'active_rate':rate_data[2][rate_data[2] != 0].reshape((-1))})
    sio.savemat('ICIC_passive_rate.mat', {'ICIC_passive_rate':rate_data[3][rate_data[3] != 0].reshape((-1))})
    sio.savemat('ICIC_ideal_active_rate.mat', {'ICIC_ideal_active_rate':rate_data[4][rate_data[4] != 0].reshape((-1))})
    sio.savemat('ICIC_active_rate.mat', {'ICIC_active_rate':rate_data[5][rate_data[5] != 0].reshape((-1))})

    # rate_data = np.array(rate_data_all)
    # rate_data = np.array(rate_data)/1e6
    # # rate_arr_no_zero = rate_data_all[0]
    # # rate_arr_no_zero = rate_arr_no_zero[rate_arr_no_zero != 0]
    # sns.kdeplot(rate_data[0][rate_data[0] != 0], label='passive')
    #
    # # rate_arr_no_zero = rate_data_all[1]
    # # rate_arr_no_zero = rate_arr_no_zero[rate_arr_no_zero != 0]
    # sns.kdeplot(rate_data[1][rate_data[1] != 0], label='ideal active')
    #
    # # rate_arr_no_zero = rate_data_all[2]
    # # rate_arr_no_zero = rate_arr_no_zero[rate_arr_no_zero != 0]
    # sns.kdeplot(rate_data[2][rate_data[2] != 0], label='active')
    #
    # # rate_arr_no_zero = rate_data_all[3]
    # # rate_arr_no_zero = rate_arr_no_zero[rate_arr_no_zero != 0]
    # sns.kdeplot(rate_data[3][rate_data[3] != 0], label='ICIC passive')
    #
    # # rate_arr_no_zero = rate_data_all[4]
    # # rate_arr_no_zero = rate_arr_no_zero[rate_arr_no_zero != 0]
    # sns.kdeplot(rate_data[4][rate_data[4] != 0], label='ICIC ideal active')
    #
    # # rate_arr_no_zero = rate_data_all[5]
    # # rate_arr_no_zero = rate_arr_no_zero[rate_arr_no_zero != 0]
    # sns.kdeplot(rate_data[5][rate_data[5] != 0], label='ICIC active')
    # plt.xlim((0, 10))
    # plt.xlabel('bit rate(Mbps)')
    # plt.legend()
    # plt.show()



    # mean_rate = [2.741, 2.751, 2.747, 3.415, 3.433, 3.436]
    # HO_mean_rate = [0.448, 0.737, 0.731, 1.837, 2.138, 2.126]
    # data = [mean_rate, HO_mean_rate]
    # xlabel = ''
    # ylabel = 'Mean bit rate(Mbps)'
    # label_list = ['mean bit rate', 'HO bit rate']
    # para_list = ['passive','ideal AHO', 'AHO', 'ICIC passive', 'ICIC ideal AHO', 'ICIC AHO']
    # plot_bar(data, xlabel, ylabel, para_list, label_list)
    #
    # plt.xticks(rotation=345)
    # plt.legend()
    # plt.show()








    # '''选择UE'''
    # HOF_type = 2
    # for _UE in UE_list:
    #     if _UE.HO_state.failure_posi[HOF_type] and _UE.HO_state.failure_posi[HOF_type-1] and _UE.type == 2:
    #         break
    # UE_type = _UE.type
    # UE_type_no = _UE.type_no
    # _UE_posi = UE_posi[UE_type][:, UE_type_no]

    # '''绘制大尺度信道'''
    # _ = plot_large_channel(PARAM, Macro_Posi, [0,1,2,3,4,5,6,7,8], shadow, _UE_posi, _UE.HO_state.failure_posi)


    # '''绘制SINR'''
    # _ = plot_SINR(_UE_posi, _UE, _UE.HO_state.failure_posi)


    # HO_result = np.array(HO_result_list).transpose()
    # HO_result = [HO_result[i] for i in range(len(HO_result))]
    # para_list = ['RB={}'.format(n) for n in RB_per_UE_list]
    # para_list = ['Para Set 1']
    # label_list = ['Success', 'Failure', 'Num of Failure Repeat UE']
    # plot_bar(HO_result, 'Parameter Set', 'HO result', para_list, label_list)
