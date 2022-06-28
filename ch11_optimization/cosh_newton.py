import numpy as np
import torch
import my_plt


# 对进⾏x优化的过程可以绘制如下。
def show_trace(results, f):
	n = max(abs(min(results)), abs(max(results)))
	f_line = torch.arange(-n, n, 0.01)
	my_plt.set_figsize()
	my_plt.plot([f_line, results], [[f(x) for x in f_line], [
			f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

# 让我们看看其他问题。给定⼀个凸双曲余弦函数c，其中c为某些常数，
# 我们可以看到经过⼏次迭代后，得到了x = 0处的全局最⼩值。
c = torch.tensor(0.5)
def f(x): # O⽬标函数
	return torch.cosh(c * x)
def f_grad(x): # ⽬标函数的梯度
	return c * torch.sinh(c * x)
def f_hess(x): # ⽬标函数的Hessian
	return c**2 * torch.cosh(c * x)
def newton(eta=1):
	x = 10.0
	results = [x]
	for i in range(10):
		x -= eta * f_grad(x) / f_hess(x)
		results.append(float(x))
	print('epoch 10, x:', x)
	return results
show_trace(newton(), f)

# 如果⼆阶导数是负的，f的值可能会趋于增加。这是这个算法的致命缺陷！
c = torch.tensor(0.15 * np.pi)
def f(x): # ⽬标函数
	return x * torch.cos(c * x)
def f_grad(x): # ⽬标函数的梯度
	return torch.cos(c * x) - c * x * torch.sin(c * x)
def f_hess(x): # ⽬标函数的Hessian
	return - 2 * c * torch.sin(c * x) - x * c**2 * torch.cos(c * x)
show_trace(newton(), f)

show_trace(newton(0.5), f)
