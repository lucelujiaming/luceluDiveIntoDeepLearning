import torch

import my_plt
import get_data_ch11
import train_framework

# Adam算法将前面所有这些技术汇总到⼀个⾼效的学习算法中。
# 不出预料，作为深度学习中使⽤的更强⼤和有效的优化算法之⼀，它⾮常受欢迎。
# 但是，有时Adam算法可能由于⽅差控制不良⽽发散。
# 在完善⼯作中，给Adam算法提供了⼀个称为Yogi的热补丁来解决这些问题。
# 下⾯我们了解⼀下Adam算法。

# 从头开始实现Adam算法并不难。
# 为⽅便起⻅，我们将时间步t存储在hyperparams字典中。
# 除此之外，⼀切都很简单。
def init_adam_states(feature_dim):
	v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
	s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
	return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
	# RMSProp算法中两项的组合都⾮常简单。
	# 最后，明确的学习率η使我们能够控制步⻓来解决收敛问题。
	beta1, beta2, eps = 0.9, 0.999, 1e-6
	for p, (v, s) in zip(params, states):
		with torch.no_grad():
			# 参见(11.10.1)
			# 动量和规模在状态变量中清晰可⻅，它们相当独特的定义使我们移除偏项。
			v[:] = beta1 * v + (1 - beta1) * p.grad
			# 参见(11.10.1)
			s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
			# 参见(11.10.2)
			v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
			s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
			# 参见(11.10.3)
			p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
								+ eps)
	p.grad.data.zero_()
	hyperparams['t'] += 1
# 现在，我们⽤以上Adam算法来训练模型，这⾥我们使⽤η = 0.01的学习率。
data_iter, feature_dim = get_data_ch11.get_data_ch11(batch_size=10)
train_framework.train_ch11(adam, init_adam_states(feature_dim),
			{'lr': 0.01, 't': 1}, data_iter, feature_dim)
# 我们可以⽤深度学习框架⾃带算法应⽤Adam算法，这⾥我们只需要传递配置参数。
trainer = torch.optim.Adam
train_framework.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)

# 11.10.3 Yogi
# Adam算法也存在⼀些问题：即使在凸环境下，当st的⼆次矩估计值爆炸时，它可能⽆法收敛。
# 论⽂中，作者还进⼀步建议⽤更⼤的初始批量来初始化动量，⽽不仅仅是初始的逐点估计。
def yogi(params, states, hyperparams):
	beta1, beta2, eps = 0.9, 0.999, 1e-3
	for p, (v, s) in zip(params, states):
		with torch.no_grad():
			# 参见(11.10.1)
			v[:] = beta1 * v + (1 - beta1) * p.grad
			# 参见(11.10.1)
			# ⼀个有效的解决⽅法是将gt^2 − st−1替换为gt^2 ⊙ sgn(gt^2 − st−1)。
			# 这就是Yogi更新，现在更新的规模不再取决于偏差的量。
			# 也就是这句代码: 
			#	torch.sign(torch.square(p.grad) - s) 
			s[:] = s + (1 - beta2) * \
					torch.sign(torch.square(p.grad) - s) * \
					torch.square(p.grad)
			# 参见(11.10.2)
			v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
			s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
			# 参见(11.10.3)
			p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
					+ eps)
	p.grad.data.zero_()
	hyperparams['t'] += 1
data_iter, feature_dim = get_data_ch11.get_data_ch11(batch_size=10)
train_framework.train_ch11(yogi, init_adam_states(feature_dim),
			{'lr': 0.01, 't': 1}, data_iter, feature_dim);




