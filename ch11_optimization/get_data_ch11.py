import numpy as np
import torch
from torch import nn

import my_download
import train_framework

# 让我们来看看如何从数据中有效地⽣成⼩批量。
# 下⾯我们使⽤NASA开发的测试机翼的数据集不同⻜⾏器产⽣的噪声来⽐较这些优化算法。
#@save
my_download.DATA_HUB['airfoil'] = \
	(my_download.DATA_URL + 'airfoil_self_noise.dat', 
		'76e5be1548fd8222e5074cf0faae75edff8cf93f')

# 为⽅便起⻅，我们只使⽤前1, 500样本。数据已作预处理：
#@save
def get_data_ch11(batch_size=10, n=1500):
	data = np.genfromtxt(my_download.download('airfoil'),
					dtype=np.float32, delimiter='\t')
	# 我们移除了均值并将⽅差重新缩放到每个坐标为1。
	data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
	data_iter = train_framework.load_array((data[:n, :-1], data[:n, -1]),
				batch_size, is_train=True)
	return data_iter, data.shape[1]-1
