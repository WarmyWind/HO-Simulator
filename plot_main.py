from visualization import *
import seaborn as sns

def save_normed_shadow(shadow, nBS, sigma, save_path):
    temp_shadow = np.reshape(shadow, (nBS, -1))
    temp_shadow = np.swapaxes(temp_shadow, 0, 1)
    mean = np.mean(temp_shadow, axis=0)
    std = np.std(temp_shadow, axis=0)
    temp_shadow = (temp_shadow - mean) * sigma / std
    temp_shadow = np.swapaxes(temp_shadow, 0, 1)
    temp_shadow = np.reshape(temp_shadow, shadow.shape)

    sio.savemat(save_path, {index: np.array(temp_shadow)})

if __name__ == '__main__':
    from simulator import *
    from network_deployment import *

    PARAM = Parameter()
    cmap = 'seismic'

    width, height = 1150, 225
    origin_x = -75
    origin_y = -75
    resolution = 0.5

    sigmaX = 6
    dCov = 100
    large_h_path = 'large_h_normed_{}sigma_{}dCov_map.npy'.format(sigmaX, dCov)

    '''从文件读取阴影衰落'''
    # shadow_filepath_0 = 'ShadowFad/0520_ShadowFad_dB_{}sigmaX_{}dCov.mat'.format(sigmaX, dCov)
    shadow_filepath = 'ShadowFad/0523_ShadowFad_dB_normed_{}sigmaX_{}dCov.mat'.format(sigmaX, dCov)
    index = 'shadowFad_dB'

    # shadowFad_dB_0 = get_shadow_from_mat(shadow_filepath_0, index)
    shadowFad_dB = get_shadow_from_mat(shadow_filepath, index)
    # shadowFad_dB = np.zeros(shadowFad_dB.shape)

    # sns.kdeplot(np.reshape(shadowFad_dB_0, (-1)), label='raw')
    # sns.kdeplot(np.reshape(shadowFad_dB, (-1)), label='normed')
    # plt.legend()
    # plt.show()

    # '''生成和保存归一化后的阴影衰落'''
    # shadow_save_path = 'ShadowFad/0523_ShadowFad_dB_normed_{}sigmaX_{}dCov.mat'.format(sigmaX, dCov)
    # save_normed_shadow(shadowFad_dB, nBS=8, sigma=sigmaX, save_path=shadow_save_path)


    num_plot = 1
    plt.figure(figsize=(9, 3))
    # grid = plt.GridSpec(1, 4*(num_plot) + 1, wspace=0.5, hspace=0.5)
    grid = plt.GridSpec(9, 3)
    axes = []

    _ax = plt.subplot(grid[:8, :])
    axes.append(_ax)

    last_ax = plt.subplot(grid[-1, :])
    axes.append(last_ax)
    ax = axes[0]

    '''生成BS位置'''
    Macro_Posi = road_cell_struct(PARAM.nCell, PARAM.Dist)




    '''初始化信道、服务信息'''
    shadow = ShadowMap(shadowFad_dB)

    def plot_hot_map(map, color_norm, cmap, ax=None, loc='best'):
        if ax == None:
            fig, ax = plt.subplots()
        ax.imshow(map.transpose(), norm=color_norm, cmap=cmap, origin='lower')
        # ax.legend(loc=loc)

        return ax


    if os.path.isfile(large_h_path):
        large_h_map = np.load(large_h_path, allow_pickle=True)

    else:


        large_h_map = np.zeros((PARAM.nCell,len(np.arange(0, width, resolution)), len(np.arange(0, height, resolution))))
        SINR_dB_map = np.zeros((len(np.arange(0, width, resolution)), len(np.arange(0, height, resolution))))
        SNR_dB_map = np.zeros((len(np.arange(0, width, resolution)), len(np.arange(0, height, resolution))))

        for x in range(large_h_map.shape[1]):
            x_posi = x * resolution + resolution / 2 + origin_x
            # _temp_x_map = []
            for y in range(large_h_map.shape[2]):
                y_posi = y*resolution + resolution / 2 + origin_y
                _UE_posi = x_posi + 1j*y_posi
                for _BS_no in range(len(Macro_Posi)):
                    _BS_posi = Macro_Posi[_BS_no]
                    _large_fading_dB = get_large_fading_dB_from_posi(PARAM, _UE_posi, _BS_posi, _BS_no, shadow, BS_type='Macro', scene=0)
                    large_h_map[_BS_no, x, y] = 10 ** (-_large_fading_dB/20)

        np.save(large_h_path, large_h_map)


    # _temp_h_map = large_h_map[:, x, y]
    _temp_h_map = np.sort(-large_h_map, axis=0)[:2, ...]
    _best_h = -_temp_h_map[0, ...]
    _second_best_h = -_temp_h_map[1, ...]
    _h_power_diff_dB_map = 10 * np.log10(_best_h / _second_best_h)
    # _h_abs_diff_map = np.abs(_best_h - _second_best_h)
    # _h_power_abs_diff_map = np.abs(np.square(_best_h) - np.square(_second_best_h))
    # _h_power_abs_diff_dB_map = 10 * np.log10(_h_power_abs_diff_map)

    _best_BS = np.argmax(large_h_map, axis=0)


    _signal_power = np.square(_best_h)
    _interf_power = np.sum(np.square(large_h_map), axis=0) - _signal_power
    _SINR = _signal_power / (_interf_power + PARAM.sigma2)
    _SNR = _signal_power / PARAM.sigma2
    SINR_dB_map = 10 * np.log10(_SINR)
    SNR_dB_map = 10 * np.log10(_SNR)


    '''plot hot map'''
    # hot_map_data = 10 * np.log10(_signal_power[::2, ::2])
    hot_map_data = SNR_dB_map[::2, ::2]

    # map_max = np.max(hot_map_data)
    # map_max = -20
    # map_max = 1e-6
    # map_max = 30
    map_max = 80

    # map_min = np.min(hot_map_data)
    map_min = 0
    # map_min = -100
    norm = mpl.colors.Normalize(vmin=map_min, vmax=map_max)
    ax = plot_hot_map(hot_map_data, norm, cmap, ax)

    '''绘制BS位置'''
    Macro_Posi = Macro_Posi - origin_x - 1j * origin_y
    ax = plot_BS_location(Macro_Posi, ax=ax)
    dist = np.abs(Macro_Posi[0] - Macro_Posi[1])
    ax = plot_hexgon(ax, Macro_Posi, dist)
    ax.axhline(20-origin_y, c='black', ls='-', lw=1)
    ax.axhline(20-origin_y + PARAM.RoadWidth, c='black', label='road', ls='-', lw=1)
    ax.set_xlim(0,width)
    ax.set_ylim(0,height)
    ax = axes[1]
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal')
    plt.tight_layout()
    plt.show()

    # for i in range(num_plot):
    #     ax = axes[i]
    #     map = map_list[i]
    #     ax.imshow(map.transpose(), norm=norm, cmap='Blues', origin='lower')
    #     _BS_x = (np.real(BS_posi) - np.min(x)) / interval_x
    #     _BS_y = (np.imag(BS_posi) - np.min(y)) / interval_y
    #     # ax.scatter(_BS_x, _BS_y, c='r', label='BS')
    #     ax.legend(loc=loc)
    #     # ax.set_xlabel(xlabel)
    #     # ax.set_ylabel(ylabel)
    #     add_scalr_bar(ax, 17.5, 20, 0, interval_x * 2.5, fineness)
    #     add_scalr_bar(ax, 0, 2.5, 0, interval_y * 2.5, fineness, 'horizontal')
    #     ax.set_xticks(np.arange(0, fineness + 1, fineness / 8))
    #     ax.set_yticks(np.arange(0, fineness + 1, fineness / 8))
    #     ax.set_title(title_list[i])
    #
    # # 展示colorbar
    # ax = axes[-1]
    # # norm = mpl.colors.Normalize(vmin=map_rate_min, vmax=map_rate_max)
    # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Blues'),
    #              cax=ax, orientation='vertical', label='Bit Rate')
    # # 避免图片显示不完全
    # plt.tight_layout()
    # # plt.show()