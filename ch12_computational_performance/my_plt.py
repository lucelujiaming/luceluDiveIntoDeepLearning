import torch
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt

# Set format
def use_svg_display():  #@save
	"""使⽤svg格式在Jupyter中显⽰绘图"""
	backend_inline.set_matplotlib_formats('svg')

# 设置图表⼤⼩。
def set_figsize(figsize=(3.5, 2.5)): #@save
	"""设置matplotlib的图表⼤⼩"""
	use_svg_display()
	plt.rcParams['figure.figsize'] = figsize

# 设置由matplotlib⽣成图表的轴的属性。
def set_axes(axes, xlabel, ylabel, 
		xlim, ylim, xscale, yscale, legend): #@save
	"""设置matplotlib的轴"""
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel)
	axes.set_xscale(xscale)
	axes.set_yscale(yscale)
	axes.set_xlim(xlim)
	axes.set_ylim(ylim)
	if legend:
		axes.legend(legend)
	axes.grid()

# 我们定义了plot函数来简洁地绘制多条曲线。
# 包含12个参数。
#    X - X坐标点。
#    Y - Y坐标点。值得注意的是，这里Y是一个列表。可以包括多条曲线的Y值。
#    xlabel，ylabel - XY坐标的标签字符串。
#    legend - 这个是一个字符串列表，和Y坐标点代表的曲线一一对应。给出每一条曲线的业务含义。
#    xlim，ylim - 沿x、y（和z轴，如果是3D图）轴显示的变量的最小值和最大值限制。
#    xscale，yscale - 设置x轴/y轴比例。
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
		ylim=None, xscale='linear', yscale='linear',
		fmts=('-', 'm--', 'g-.', 'r:'), 
		figsize=(3.5, 2.5), axes=None):  #@save
	"""绘制数据点"""
	if legend is None:
		legend = []

	set_figsize(figsize)
	axes = axes if axes else plt.gca()

	# 如果X有⼀个轴，输出True
	def has_one_axis(X):
		return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
				and not hasattr(X[0], "__len__"))
	if has_one_axis(X):
		X = [X]
	if Y is None:
		X, Y = [[]] * len(X), X
	elif has_one_axis(Y):
		Y = [Y]

	if len(X) != len(Y):
		X = X * len(Y)
	axes.cla()
	for x, y, fmt in zip(X, Y, fmts):
		if len(x):
			axes.plot(x, y, fmt)
		else:
			axes.plot(y, fmt)
	set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
	# 如果在Jupyter中运行，就不需要这句。如果单独执行，就需要加上。
	plt.show()

# From test_gd
# 多元梯度下降
# 我们还需要两个辅助函数：
#     第⼀个是update函数，并将其应⽤于初始值20次；
#     第⼆个函数会显⽰x的轨迹。
def train_2d(trainer, steps=20, f_grad=None): #@save
	"""⽤定制的训练机优化2D⽬标函数"""
	# s1和s2是稍后将使⽤的内部状态变量
	x1, x2, s1, s2 = -5, -2, 0, 0
	results = [(x1, x2)]
	for i in range(steps):
		if f_grad:
			x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
		else:
			x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
		results.append((x1, x2))
	print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
	return results    
def show_trace_2d(f, results): #@save
	"""显⽰优化过程中2D变量的轨迹"""
	set_figsize()
	plt.plot(*zip(*results), '-o', color='#ff7f0e')
	x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
					torch.arange(-3.0, 1.0, 0.1))
	plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()
