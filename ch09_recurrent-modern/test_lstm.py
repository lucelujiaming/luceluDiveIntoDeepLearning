import torch
from torch import nn

import train_framework
import my_data_time_machine
# ⾸先加载时光机器数据集。
batch_size, num_steps = 32, 35
train_iter, vocab = my_data_time_machine.load_data_time_machine(batch_size, num_steps)

# 要定义和初始化模型参数。
# 如前所述，超参数num_hiddens定义隐藏单元的数量。
def get_lstm_params(vocab_size, num_hiddens, device):
	num_inputs = num_outputs = vocab_size
	# 我们按照标准差0.01的⾼斯分布初始化权重，并将偏置项设为0。
	def normal(shape):
		return torch.randn(size=shape, device=device)*0.01
	def three():
		return (normal((num_inputs, num_hiddens)),
	
	normal((num_hiddens, num_hiddens)),
	torch.zeros(num_hiddens, device=device))
	W_xi, W_hi, b_i = three() # 输⼊⻔参数
	W_xf, W_hf, b_f = three() # 遗忘⻔参数
	W_xo, W_ho, b_o = three() # 输出⻔参数
	W_xc, W_hc, b_c = three() # 候选记忆元参数
	# 输出层参数
	W_hq = normal((num_hiddens, num_outputs))
	b_q = torch.zeros(num_outputs, device=device)
	# 附加梯度
	params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
						b_c, W_hq, b_q]
	for param in params:
		param.requires_grad_(True)
	return params

# ⻓短期记忆⽹络的隐状态需要返回⼀个额外的记忆元，
def init_lstm_state(batch_size, num_hiddens, device):
	# 单元的值为0，形状为（批量⼤⼩，隐藏单元数）。
	return (torch.zeros((batch_size, num_hiddens), device=device),
		torch.zeros((batch_size, num_hiddens), device=device))

# 实际模型的定义与我们前⾯讨论的⼀样：提供三个⻔和⼀个额外的记忆元。
# 请注意，只有隐状态才会传递到输出层，⽽记忆元Ct不直接参与输出计算。
def lstm(inputs, state, params):
	[W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
			W_hq, b_q] = params
	(H, C) = state
	outputs = []
	for X in inputs:
		# 公式(9.2.1)
		I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
		F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
		O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
		# 使用公式(9.2.2)计算候选记忆元。
		C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
		# 使用公式(9.2.3)计算记忆元。
		C = F * C + I * C_tilda
		# 使用使用公式(9.2.4)计算隐状态。
		H = O * torch.tanh(C)
		# 有了H，我们就可以计算Y了。
		Y = (H @ W_hq) + b_q
		outputs.append(Y)
	return torch.cat(outputs, dim=0), (H, C)

vocab_size, num_hiddens, device = len(vocab), 256, train_framework.try_gpu()
num_epochs, lr = 500, 1
# 通过实例化 8.5节中引⼊的RNNModelScratch类来训练⼀个⻓短期记忆⽹络。
model = train_framework.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
init_lstm_state, lstm)
train_framework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

