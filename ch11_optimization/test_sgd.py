import math
import torch
import test_gd

def show_trace_2d(f, results): #@save
	"""显⽰优化过程中2D变量的轨迹"""
	my_plt.set_figsize()
	my_plt.plt.plot(*zip(*results), '-o', color='#ff7f0e')
	x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
					torch.arange(-3.0, 1.0, 0.1))
	my_plt.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
	my_plt.plt.xlabel('x1')
	my_plt.plt.ylabel('x2')
	my_plt.plt.show()

# 我们将把它与梯度下降进⾏⽐较，
def f(x1, x2): # ⽬标函数
	return x1 ** 2 + 2 * x2 ** 2
def f_grad(x1, x2): # ⽬标函数的梯度
	return 2 * x1, 4 * x2
def sgd(x1, x2, s1, s2, f_grad):
	g1, g2 = f_grad(x1, x2)
	# 模拟有噪声的梯度
	# ⽅法是向梯度添加均值为0、⽅差为1的随机噪声，以模拟随机梯度下降。
	g1 += torch.normal(0.0, 1, (1,))
	g2 += torch.normal(0.0, 1, (1,))
	eta_t = eta * lr()
	return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
	return 1
eta = 0.1
lr = constant_lr # 常数学习速度
test_gd.show_trace_2d(f, test_gd.train_2d(sgd, steps=50, f_grad=f_grad))

# 让我们看看通过指数衰减（exponential decay）来更积极地减低学习率。
def exponential_lr():
	# 在函数外部定义，⽽在内部更新的全局变量
	global t 
	t += 1
	return math.exp(-0.1 * t)
t = 1
lr = exponential_lr
test_gd.show_trace_2d(f, test_gd.train_2d(sgd, steps=1000, f_grad=f_grad))

# 如果我们使⽤多项式衰减，其中学习率随迭代次数的平⽅根倒数衰减，那么仅在50次迭代之后，收敛就会更好。
def polynomial_lr():
	# 在函数外部定义，⽽在内部更新的全局变量
	global t 
	t += 1
	return (1 + 0.1 * t) ** (-0.5) 
t = 1
lr = polynomial_lr
test_gd.show_trace_2d(f, test_gd.train_2d(sgd, steps=50, f_grad=f_grad))



