import numpy as np
import torch
from mpl_toolkits import mplot3d
import my_plt

# 为了说明这⼀点，让我们绘制⼀些函数并检查哪些函数满⾜要求。
# 下⾯我们定义⼀些函数，包括凸函数和⾮凸函数。
f = lambda x: 0.5 * x**2 # 凸函数
g = lambda x: torch.cos(np.pi * x) # ⾮凸函数
h = lambda x: torch.exp(0.5 * x) # 凸函数
x, segment = torch.arange(-2, 2, 0.01), torch.tensor([-1.5, 1])
my_plt.use_svg_display()
_, axes = my_plt.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
	my_plt.plot([x, segment], [func(x), func(segment)], axes=ax)
# 例如，对于凸函数f(x) = (x − 1)^2，有⼀个局部最⼩值x = 1，它也是全局最⼩值。
f = lambda x: (x - 1) ** 2
my_plt.set_figsize()
my_plt.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')

