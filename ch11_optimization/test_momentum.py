import torch

import my_plt
import get_data_ch11
import train_framework

eta = 0.4
# 我们在 11.3节中使⽤了f(x) = x1^2 + 2 * x2^2，即中度扭曲的椭球⽬标。
# 我们通过向x1⽅向伸展它来进⼀步扭曲这个函数。
#        f(x) = 0.1 * x1^2 + 2 * x2^2
def f_2d(x1, x2):
	return 0.1 * x1 ** 2 + 2 * x2 ** 2
# 下面定义f(x)的导数。
def gd_2d(x1, x2, s1, s2):
	return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

if __name__ == '__main__':
	# 让我们看看在这个新函数上执⾏梯度下降时会发⽣什么。
	my_plt.show_trace_2d(f_2d, my_plt.train_2d(gd_2d))

	# 下⾯的例⼦说明了即使学习率从0.4略微提⾼到0.6，也会发⽣变化。
	# x1⽅向上的收敛有所改善，但整体来看解的质量更差了。
	eta = 0.6
	my_plt.show_trace_2d(f_2d, my_plt.train_2d(gd_2d))

# 动量法（momentum）使我们能够解决上⾯描述的梯度下降问题。
def momentum_2d(x1, x2, v1, v2):
	v1 = beta * v1 + 0.2 * x1
	v2 = beta * v2 + 4 * x2
	return x1 - eta * v1, x2 - eta * v2, v1, v2

if __name__ == '__main__':
	# 让我们快速看⼀下动量法（momentum）算法在实验中的表现如何。
	eta, beta = 0.6, 0.5
	my_plt.show_trace_2d(f_2d, my_plt.train_2d(momentum_2d))
	# 它⽐没有动量时解将会发散要好得多。
	eta, beta = 0.6, 0.25
	my_plt.show_trace_2d(f_2d, my_plt.train_2d(momentum_2d))

	# 不同于在梯度下降或者随机梯度下降中取步⻓η，我们选取步⻓: η /(1−β)，
	# 同时处理潜在表现可能会更好的下降⽅向。这是集两种好处于⼀⾝的做法。
	my_plt.set_figsize()
	betas = [0.95, 0.9, 0.6, 0]
	for beta in betas:
		x = torch.arange(40).detach().numpy()
	my_plt.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
	my_plt.plt.xlabel('time')
	my_plt.plt.legend();

# 首先初始化为零。
def init_momentum_states(feature_dim):
	v_w = torch.zeros((feature_dim, 1))
	v_b = torch.zeros(1)
	return (v_w, v_b)
# 相⽐于⼩批量随机梯度下降，动量⽅法需要维护⼀组辅助变量，即速度。
# 它与梯度以及优化问题的变量具有相同的形状。在下⾯的实现中，我们称这些变量为states。
def sgd_momentum(params, states, hyperparams):
	for p, v in zip(params, states):
		with torch.no_grad():
			v[:] = hyperparams['momentum'] * v + p.grad
			p[:] -= hyperparams['lr'] * v 
	p.grad.data.zero_()

# 让我们看看它在实验中是如何运作的。
def train_momentum(lr, momentum, num_epochs=2):
	train_framework.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
			{'lr': lr, 'momentum': momentum}, data_iter,
			feature_dim, num_epochs)

if __name__ == '__main__':
	# 让我们看看它在实验中是如何运作的。
	data_iter, feature_dim = get_data_ch11.get_data_ch11(batch_size=10)
	train_momentum(0.02, 0.5)
	# 当我们将动量超参数momentum增加到0.9时，它相当于有效样本数量增加到1/(1 - 0.9) = 10。
	# 我们将学习率略微降⾄0.01，以确保可控。
	train_momentum(0.01, 0.9)
	# 降低学习率进⼀步解决了任何⾮平滑优化问题的困难，将其设置为0.005会产⽣良好的收敛性能。
	train_momentum(0.005, 0.9)

if __name__ == '__main__':
	# 由于深度学习框架中的优化求解器早已构建了动量法，设置匹配参数会产⽣⾮常类似的轨迹。
	trainer = torch.optim.SGD
	train_framework.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)

if __name__ == '__main__':
	# 鉴于上述结果，让我们看看当我们最⼩化函数f(x) = (λ/2) * x^2时会发⽣什么。
	# 显⽰了在我们将学习率η提⾼到ηλ = 1之前，收敛率最初是如何提⾼的。
	# 超过该数值之后，梯度开始发散，对于ηλ > 2⽽⾔，优化问题将会发散。
	lambdas = [0.1, 1, 10, 19]
	eta = 0.1
	my_plt.set_figsize((6, 4))
	for lam in lambdas:
		t = torch.arange(20).detach().numpy()
	my_plt.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
	my_plt.plt.xlabel('time')
	my_plt.plt.legend();
	my_plt.plt.show()
