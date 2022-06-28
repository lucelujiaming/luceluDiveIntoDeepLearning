import math
import time
import numpy as np
import torch
import my_plt

def normal(x, mu, sigma):
	p = 1 / math.sqrt(2 * math.pi * sigma**2)
	return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 再次使⽤numpy进⾏可视化
x = np.arange(-7, 7, 0.01) 

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
my_plt.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
		ylabel='p(x)', figsize=(4.5, 2.5),
		legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])

