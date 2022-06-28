import math
import torch

import my_plt
import get_data_ch11
import train_framework

# 为了获得良好的准确性，我们⼤多希望在训练的过程中降低学习率。
# 解决此问题的⼀个⽅法是记录我们看到特定特征的次数，然后将其⽤作调整学习率。
# AdaGrad算法通过将粗略的计数器s(i, t)替换为先前观察所得梯度的平⽅之和来解决这个问题。
# 它使⽤s(i, t+ 1) = s(i, t) + (∂if(x)) ^ 2来调整学习率。

def adagrad_2d(x1, x2, s1, s2):
	eps = 1e-6
	# f(x)的两个偏导数。也就是先前观察所得梯度。
	g1, g2 = 0.2 * x1, 4 * x2
	# 先前观察所得梯度的平⽅之和。
	s1 += g1 ** 2
	s2 += g2 ** 2
	# 调整学习率。
	x1 -= eta / math.sqrt(s1 + eps) * g1
	x2 -= eta / math.sqrt(s2 + eps) * g2
	return x1, x2, s1, s2

# 我们仍然以同⼀函数为例：
#    f(x) = 0.1 * x1^2 + 2 * x2^2
def f_2d(x1, x2):
	return 0.1 * x1 ** 2 + 2 * x2 ** 2
# 我们将使⽤与之前相同的学习率来实现AdaGrad算法，即η = 0.4。
# 可以看到，⾃变量的迭代轨迹较平滑。
# 但由于st的累加效果使学习率不断衰减，⾃变量在迭代后期的移动幅度较⼩。
eta = 0.4
my_plt.show_trace_2d(f_2d, my_plt.train_2d(adagrad_2d))
# 我们将学习率提⾼到2，可以看到更好的表现。
eta = 2
my_plt.show_trace_2d(f_2d, my_plt.train_2d(adagrad_2d))

# 同动量法⼀样，AdaGrad算法需要对每个⾃变量维护同它⼀样形状的状态变量。
def init_adagrad_states(feature_dim):
	s_w = torch.zeros((feature_dim, 1))
	s_b = torch.zeros(1)
	return (s_w, s_b)
def adagrad(params, states, hyperparams):
	eps = 1e-6
	for p, s in zip(params, states):
		with torch.no_grad():
			s[:] += torch.square(p.grad)
			p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
	p.grad.data.zero_()
# 这⾥使⽤更⼤的学习率来训练模型。
data_iter, feature_dim = get_data_ch11.get_data_ch11(batch_size=10)
train_framework.train_ch11(adagrad, init_adagrad_states(feature_dim),
				{'lr': 0.1}, data_iter, feature_dim);

# 我们可直接使⽤深度学习框架中提供的AdaGrad算法来训练模型。
trainer = torch.optim.Adagrad
train_framework.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)

