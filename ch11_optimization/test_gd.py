import numpy as np
import torch
import my_plt

def f(x): # ⽬标函数
	return x ** 2
def f_grad(x): # ⽬标函数的梯度(导数)
	return 2 * x

def gd(eta, f_grad):
	x = 10.0
	results = [x]
	for i in range(10):
		x -= eta * f_grad(x)
		results.append(float(x))
	print(f'epoch 10, x: {x:f}')
	return results

if __name__ == '__main__':
	results = gd(0.2, f_grad)

# 对进⾏x优化的过程可以绘制如下。
def show_trace(results, f):
	n = max(abs(min(results)), abs(max(results)))
	f_line = torch.arange(-n, n, 0.01)
	my_plt.set_figsize()
	my_plt.plot([f_line, results], [[f(x) for x in f_line], [
			f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

if __name__ == '__main__':
	show_trace(results, f)
	# 例如，考虑同⼀优化问题中η = 0.05的进度。
	# 如下所⽰，尽管经过了10个步骤，我们仍然离最优解很远。
	show_trace(gd(0.05, f_grad), f)
	# 相反，如果我们使⽤过⾼的学习率。在这种情况下，x的迭代不能保证降低f(x)的值。
	show_trace(gd(1.1, f_grad), f)

# 下⾯的例⼦说明了（不切实际的）⾼学习率如何导致较差的局部最⼩值。
c = torch.tensor(0.15 * np.pi)
def f(x): # ⽬标函数
	return x * torch.cos(c * x)
def f_grad(x): # ⽬标函数的梯度
	return torch.cos(c * x) - c * x * torch.sin(c * x)
if __name__ == '__main__':
	show_trace(gd(2, f_grad), f)

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
	my_plt.set_figsize()
	my_plt.plt.plot(*zip(*results), '-o', color='#ff7f0e')
	x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
					torch.arange(-3.0, 1.0, 0.1))
	my_plt.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
	my_plt.plt.xlabel('x1')
	my_plt.plt.ylabel('x2')
	my_plt.plt.show()

if __name__ == '__main__':
	# 接下来，我们观察学习率η = 0.1时优化变量x的轨迹。
	def f_2d(x1, x2): # ⽬标函数
		return x1 ** 2 + 2 * x2 ** 2
	def f_2d_grad(x1, x2): # ⽬标函数的梯度
		return (2 * x1, 4 * x2)
	def gd_2d(x1, x2, s1, s2, f_grad):
		g1, g2 = f_grad(x1, x2)
		return (x1 - eta * g1, x2 - eta * g2, 0, 0)
	eta = 0.1
	show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))




