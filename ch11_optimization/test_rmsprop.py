import math
import torch

import my_plt
import get_data_ch11
import train_framework

# [Tieleman & Hinton, 2012]建议以RMSProp算法作为将速率调度与坐标⾃适应学习率分离的简单修复⽅法。
# 问题在于，Adagrad算法将梯度gt的平⽅累加成状态⽮量st = st−1 + gt^2。
# 因此，由于缺乏规范化，没有约束⼒，st持续增⻓，⼏乎上是在算法收敛时呈线性递增。
# 解决此问题的⼀种⽅法是使⽤st/t。对于gt的合理分布来说，它将收敛。

# 让我们图像化各种数值的γ在过去40个时间步⻓的权重。
my_plt.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
	x = torch.arange(40).detach().numpy()
	my_plt.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
my_plt.plt.xlabel('time');

# 和之前⼀样，我们使⽤⼆次函数f(x) = 0.1x21+2x22来观察RMSProp算法的轨迹。
def rmsprop_2d(x1, x2, s1, s2):
	# f(x)的两个偏导数。也就是先前观察所得梯度。
	g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
	# 参见(11.8.1)
	s1 = gamma * s1 + (1 - gamma) * g1 ** 2
	s2 = gamma * s2 + (1 - gamma) * g2 ** 2
	# 解决此问题的⼀种⽅法是使⽤st/t。
	x1 -= eta / math.sqrt(s1 + eps) * g1
	x2 -= eta / math.sqrt(s2 + eps) * g2
	return x1, x2, s1, s2

# 我们仍然以同⼀函数为例：
#    f(x) = 0.1 * x1^2 + 2 * x2^2
def f_2d(x1, x2):
	return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
my_plt.show_trace_2d(f_2d, my_plt.train_2d(rmsprop_2d))

# 接下来，我们在深度⽹络中实现RMSProp算法
def init_rmsprop_states(feature_dim):
	s_w = torch.zeros((feature_dim, 1))
	s_b = torch.zeros(1)
	return (s_w, s_b)
def rmsprop(params, states, hyperparams):
	gamma, eps = hyperparams['gamma'], 1e-6
	for p, s in zip(params, states):
		# 获得梯度。
		with torch.no_grad():
			# 参见(11.8.1)
			s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
			p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
	p.grad.data.zero_()

# 我们将初始学习率设置为0.01，加权项γ设置为0.9。
# 也就是说，s累加了过去的1/(1 − γ) = 10次平⽅梯度观测值的平均值。
data_iter, feature_dim = get_data_ch11.get_data_ch11(batch_size=10)
train_framework.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
		{'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);

# 我们可直接使⽤深度学习框架中提供的RMSProp算法来训练模型。
trainer = torch.optim.RMSprop
train_framework.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
			data_iter)
