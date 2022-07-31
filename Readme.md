# HOSimulator仿真器

## 简介

HOSimulator可实现在场景中对切换和干扰协调的仿真，得到数据率、切换次数和切换成功率等结果。



## 仿真器依赖环境

python==3.7.1

matplotlib==3.4.2
numpy==1.21.1
pandas==1.3.3
scipy==1.6.2
seaborn==0.11.1
tensorflow==2.9.1
tensorflow_gpu==2.9.1
tf_slim==1.1.0
torch==1.9.0



## 模块说明

仿真器的主入口为**simulator.py**，直接运行即可根据仿真参数开始仿真。保存的仿真结果包括每一帧每个UE的数据率、各个小区中的UE列表、未服务的UE列表、GBR用户列表和干扰和接收信号功率等。根据保存的UE列表中的信息，还可以统计切换次数及成功率等。

仿真器包含以下子模块：

**para_init.py**：定义了仿真参数类**Parameter**。如果要设定仿真参数，请在该模块中的**paraset_generator**函数内进行修改。

**data_factory.py**：包含处理各种仿真数据和类型的函数。

**channel_fading.py**：包含关于衰落信道的函数。

**channel_measurement.py**：包含信道测量相关函数。

**DNNModel**：包含用户大尺度信道预测的DNN模型。

**ReinforcementLearningV1**：第一版无监督学习模型。模型输入为小区内用户数，输出为给UE分配的RB数。

**ReinforcementLearningV2**：第二版无监督学习模型。模型输入为小区内用户数、ICI和正交RB数，输出为给UE分配的RB数。

**handover_process.py**：切换过程处理模块。

**info_management.py**：仿真信息管理模块。

**network_deployment.py**：网络部署模块，包含确定基站位置的函数。

**precoding.py**：预编码模块。

**radio_access.py**：无线接入模块。

**resource_allocation.py**：资源分配模块。

**SINR_calculate.py**：SINR计算模块

**user_mobility.py**：用户移动性模块，用户位置的移动直接从文件读取。

**utils.py**：包含其他一些仿真器中的工具函数。



## 仿真说明

在仿真前，确认**simulator.py**中SimConfig设置正确，包括仿真的子帧数（8*帧数），UE轨迹的数据文件等。SimConfig.save_flag设置为1则会保存仿真结果到SimConfig.root_path。

根据需要，修改**para_init.py**中的**paraset_generator**函数。设置主动切换则令PARAM0.active_HO = True。要利用无监督学习模型做动态ICIC则令PARAM0.ICIC.dynamic = True并且PARAM0.dynamic_nRB_per_UE = True。做传统ICIC，则令PARAM0.ICIC.dynamic = False；PARAM0.dynamic_nRB_per_UE = False，并且设置PARAM0.ICIC.RB_for_edge_ratio和PARAM0.RB_per_UE（注：PARAM0.ICIC.RB_for_edge_ratio = 0则不做ICIC）。

如需修改基站位置，则在**network_deployment.py**修改，或者编写新的基站位置生成函数。

如需修改其他仿真参数，均在**para_init.py**寻找对应项进行修改即可。