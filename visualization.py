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


def plot_BS_location(Macro_Posi, Micro_Posi):
    fig, ax = plt.subplots()
    ax.scatter(np.real(Macro_Posi), np.imag(Macro_Posi), label='Macro BS')
    ax.scatter(np.real(Micro_Posi), np.imag(Micro_Posi), label='Micro BS')
    plt.legend(loc='upper right')
    plt.show()


def plot_UE_trajectory(Macro_Posi, UE_tra):
    fig, ax = plt.subplots()
    ax.scatter(np.real(Macro_Posi), np.imag(Macro_Posi), label='Macro BS')
    for i in range(UE_tra.shape[-1]):
        ax.plot(np.real(UE_tra[:, i]), np.imag(UE_tra[:, i]), label='User{}'.format(i))

    plt.legend(loc='upper right')
    plt.show()


def plot_cdf(data, xlabel, ylabel, label_list, normed=1, loc='lower right'):
    # data is a list of array
    fig, ax = plt.subplots()
    for i in range(len(data)):
        _d = data[i].flatten()
        ax.hist(_d, bins=250, density=normed, cumulative='Ture', histtype='step', label=label_list[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xbound(np.min(data), np.max(data))
    ax.set_ybound(0, 1)
    fix_hist_step_vertical_line_at_end(ax)
    plt.legend(loc=loc)
    plt.show()


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


def plot_bar(data, xlabel, ylabel, para_list, label_list, loc='upper left'):
    # 创建分组柱状图，需要自己控制x轴坐标
    xticks = np.arange(len(para_list))

    fig, ax = plt.subplots()
    # 所有门店第一种产品的销量，注意控制柱子的宽度，这里选择0.25
    for i in range(len(label_list)):
        ax.bar(xticks + i * 0.25, data[i], width=0.25, label=label_list[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    # 最后调整x轴标签的位置
    ax.set_xticks(xticks + 0.25)
    ax.set_xticklabels(para_list)

    plt.legend(loc=loc)
    plt.show()


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

        for nDrop in range(UE_posi.shape[0]):
            for _UE in range(UE_posi.shape[1]):
                _idx_x, _idx_y = trans_to_map_index(UE_posi[nDrop, _UE])

                map[_idx_x, _idx_y] = map[_idx_x, _idx_y] + rate_data[i][nDrop, _UE]
                map_cnt[_idx_x, _idx_y] = map_cnt[_idx_x, _idx_y] + 1

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
    plt.show()


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
