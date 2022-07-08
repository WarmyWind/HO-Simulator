from visualization import *
'''保存路径和字典索引'''
save_path = '7BS_outer_itf_map.mat'
index = 'itf_map'

'''生成BS位置'''
nCell = 7
Dist = 150
Macro_Posi = cell_struction(nCell, Dist)

'''基站和大尺度衰落参数'''
PtmaxdBm = 46  # 基站最大发射功率
Ptmax = 10 ** (PtmaxdBm / 10) / 1000
nRB = 30
antGaindB = 0
dFactordB = 37.6
pLoss1mdB = 15.3

'''地图参数'''
origin_x_point = -250
origin_y_point = -200
Delta_Resolution = 0.5
x_length = 500
y_length = 400
x_range = np.arange(0,x_length,Delta_Resolution)
y_range = np.arange(0,y_length,Delta_Resolution)


posi_map = np.zeros((len(y_range),len(x_range))).astype(np.complex_)
for _x_idx in range(len(x_range)):
    for _y_idx in range(len(y_range)):
        posi_map[_y_idx, _x_idx] = x_range[_x_idx]+origin_x_point + 1j*(y_range[_y_idx]+origin_y_point)

interf_map = np.zeros((len(Macro_Posi), 2, len(y_range), len(x_range)))

outer_group0_BS_posi = [np.exp(1j*(i*np.pi/3+np.pi/6))*Dist*np.sqrt(3) for i in range(6)]
outer_group1_BS_posi = [np.exp(1j*(i*2*np.pi/3))*Dist*2 for i in range(3)]
outer_group2_BS_posi = [np.exp(1j*(i*2*np.pi/3+np.pi/3))*Dist*np.sqrt(3) for i in range(3)]
outer_BS_posi = [outer_group0_BS_posi,outer_group1_BS_posi,outer_group2_BS_posi]
inner_BS_group = [1,2,0,1,2,2,1]
for _BS_idx in range(len(Macro_Posi)):
    # _BS_posi = Macro_Posi[_BS_idx]
    for ICIC_flag in range(2):
        if not ICIC_flag:
            for _group_idx in range(len(outer_BS_posi)):
                for _outer_BS_posi in outer_BS_posi[_group_idx]:
                    _dist = np.abs(posi_map - _outer_BS_posi)
                    if np.count_nonzero(_dist == 0):
                        raise Exception('Dist == 0')
                    large_fading_dB = pLoss1mdB + dFactordB * np.log10(_dist) - antGaindB
                    large_scale_channel = 10 ** (-large_fading_dB / 20)
                    _estimated_interf_power = Ptmax / nRB * np.square(large_scale_channel)
                    interf_map[_BS_idx,ICIC_flag,...] = interf_map[_BS_idx,ICIC_flag,...] + _estimated_interf_power

        else:
            for _outer_BS_posi in outer_BS_posi[inner_BS_group[_BS_idx]]:
                _dist = np.abs(posi_map - _outer_BS_posi)
                if np.count_nonzero(_dist == 0):
                    raise Exception('Dist == 0')
                large_fading_dB = pLoss1mdB + dFactordB * np.log10(_dist) - antGaindB
                large_scale_channel = 10 ** (-large_fading_dB / 20)
                _estimated_interf_power = Ptmax / nRB * np.square(large_scale_channel)
                interf_map[_BS_idx, ICIC_flag, ...] = interf_map[_BS_idx, ICIC_flag, ...] + _estimated_interf_power


sio.savemat(save_path, {index: np.array(interf_map)})
