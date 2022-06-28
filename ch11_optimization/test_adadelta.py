import torch

import my_plt
import get_data_ch11
import train_framework

# Adadelta是AdaGrad的另⼀种变体，主要区别在于前者减少了学习率适应坐标的数量。
# Adadelta需要为每个变量维护两个状态变量，即st和∆xt。这将产⽣以下实现。
def init_adadelta_states(feature_dim):
	s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
	delta_w, delta_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
	return ((s_w, delta_w), (s_b, delta_b))
def adadelta(params, states, hyperparams):
	rho, eps = hyperparams['rho'], 1e-5
	for p, (s, delta) in zip(params, states):
		with torch.no_grad():
			# In-placeupdatesvia[:]
			# 参见公式(11.9.1)
			s[:] = rho * s + (1 - rho) * torch.square(p.grad)
			# 参见公式(11.9.3)
			g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
			# 参见公式(11.9.2)
			p[:] -= g
			# 参见公式(11.9.4)
			delta[:] = rho * delta + (1 - rho) * g * g 
	p.grad.data.zero_()
# 对于每次参数更新，选择ρ = 0.9相当于10个半衰期。由此我们得到：
data_iter, feature_dim = get_data_ch11.get_data_ch11(batch_size=10)
train_framework.train_ch11(adadelta, init_adadelta_states(feature_dim),
				{'rho': 0.9}, data_iter, feature_dim);
# 为了简洁实现，我们只需使⽤Trainer类中的adadelta算法。
trainer = torch.optim.Adadelta
train_framework.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)


