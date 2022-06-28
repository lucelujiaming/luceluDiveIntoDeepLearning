import numpy as np
import torch
from mpl_toolkits import mplot3d
import my_plt

def f(x):
	return x * torch.cos(np.pi * x)
def g(x):
	return f(x) + 0.2 * torch.cos(5 * np.pi * x)

def annotate(text, xy, xytext): #@save
	my_plt.plt.gca().annotate(text, xy=xy, xytext=xytext,
		arrowprops=dict(arrowstyle='->'))
x = torch.arange(0.5, 1.5, 0.01)
my_plt.set_figsize((4.5, 2.5))
my_plt.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))

# 我们可以近似该函数的局部最⼩值和全局最⼩值。
x = torch.arange(-1.0, 2.0, 0.01)
my_plt.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))

# 鞍点也是梯度消失的另⼀个原因。
# 鞍点（saddle point）是指函数的所有梯度都消失但既不是全局最⼩值也不是局部最⼩值的任何位置。
x = torch.arange(-2.0, 2.0, 0.01)
my_plt.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))

# 如下例所⽰，较⾼维度的鞍点甚⾄更加隐蔽。
x, y = torch.meshgrid(
		torch.linspace(-1.0, 1.0, 101), torch.linspace(-1.0, 1.0, 101))
z = x**2 - y**2
ax = my_plt.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
my_plt.plt.xticks(ticks)
my_plt.plt.yticks(ticks)
ax.set_zticks(ticks)
my_plt.plt.xlabel('x')
my_plt.plt.ylabel('y');
my_plt.plt.show()

# 可能遇到的最隐蔽的问题是梯度消失。
x = torch.arange(-2.0, 5.0, 0.01)
my_plt.plot(x, [torch.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))




