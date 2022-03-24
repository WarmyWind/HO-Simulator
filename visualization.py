'''
本模块包含可视化方法:
    plot_BS_location
    plot_UE_trajectory
    plot_cdf
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

def plot_cdf(data, xlabel, ylabel, label_list, normed=1):
    # data is a list of array
    fig, ax = plt.subplots()
    for i in range(len(data)):
        _d = data[i].flatten()
        ax.hist(_d, bins = 250, density = normed, cumulative='Ture', histtype='step', label=label_list[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xbound(np.min(data),np.max(data))
    ax.set_ybound(0,1)
    fix_hist_step_vertical_line_at_end(ax)
    plt.legend(loc='lower right')
    plt.show()

def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


def plot_bar(data, xlabel, ylabel, para_list, label_list):
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

    plt.legend(loc='upper right')
    plt.show()

