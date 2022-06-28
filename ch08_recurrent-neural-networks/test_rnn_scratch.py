import math
import torch
from torch import nn
from torch.nn import functional as F
import test_text_preprocessing
import my_plt
import test_language_models_and_dataset
import train_framework
import my_timer
import my_plt

if __name__ == '__main__':
	# 加载数据集
	batch_size, num_steps = 32, 35
	# 在train_iter中，每个词元都表⽰为⼀个数字索引。
	train_iter, vocab = test_language_models_and_dataset.load_data_time_machine(batch_size, num_steps)
	# 索引为0和2的独热向量如下所⽰：
	print("F.one_hot(torch.tensor([0, 2]), len(vocab)) : ", 
		F.one_hot(torch.tensor([0, 2]), len(vocab)))

	# 我们每次采样的⼩批量数据形状是⼆维张量：（批量⼤⼩，时间步数）。
	X = torch.arange(10).reshape((2, 5))
	# one_hot函数将这样⼀个⼩批量数据转换成三维张量，
	# 张量的最后⼀个维度等于词表⼤⼩（len(vocab)）。
	print("F.one_hot(X.T, 28).shape : ", F.one_hot(X.T, 28).shape)

# 接下来，我们初始化循环神经⽹络模型的模型参数。
# 隐藏单元数num_hiddens是⼀个可调的超参数。
def get_params(vocab_size, num_hiddens, device):
	# 当训练语⾔模型时，输⼊和输出来⾃相同的词表。
	# 因此，它们具有相同的维度，即词表的⼤⼩。
	num_inputs = num_outputs = vocab_size
	# torch.randn:用来生成随机数字的tensor，
	# 这些随机数字满足标准正态分布（0~1）。
	# 参数shape表示随机数的维度。
	def normal(shape):
		return torch.randn(size=shape, device=device) * 0.01
	# 隐藏层参数
	# 使用正态分布随机数进行初始化。
	W_xh = normal((num_inputs, num_hiddens))
	W_hh = normal((num_hiddens, num_hiddens))
	b_h = torch.zeros(num_hiddens, device=device)
	# 输出层参数
	W_hq = normal((num_hiddens, num_outputs))
	b_q = torch.zeros(num_outputs, device=device)
	# 附加梯度
	params = [W_xh, W_hh, b_h, W_hq, b_q]
	# 打开梯度。
	for param in params:
		param.requires_grad_(True)
	return params

# 为了定义循环神经⽹络模型，我们⾸先需要⼀个init_rnn_state函数在初始化时返回隐状态。
def init_rnn_state(batch_size, num_hiddens, device):
	# 返回是⼀个张量，张量全⽤0填充，形状为（批量⼤⼩，隐藏单元数）。
	return (torch.zeros((batch_size, num_hiddens), device=device), )

# 下⾯的rnn函数定义了如何在⼀个时间步内计算隐状态和输出。
def rnn(inputs, state, params):
	# inputs的形状：(时间步数量，批量⼤⼩，词表⼤⼩)
	W_xh, W_hh, b_h, W_hq, b_q = params
	H, = state
	outputs = []
	# X的形状：(批量⼤⼩，词表⼤⼩)
	# 循环神经⽹络模型通过inputs最外层的维度实现循环，
	# 以便逐时间步更新⼩批量数据的隐状态H。
	for X in inputs:
		# 使⽤tanh函数作为激活函数。
		# 当元素在实数上满⾜均匀分布时，tanh函数的平均值为0。
		# 采用(8.4.5)的公式：
		#		Ht = ϕ(XtWxh + Ht-1Whh + bh). 
		H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
		Y = torch.mm(H, W_hq) + b_q
		outputs.append(Y)
	return torch.cat(outputs, dim=0), (H,)

# 定义了所有需要的函数之后，接下来我们创建⼀个类来包装这些函数，
# 并存储从零开始实现的循环神经⽹络模型的参数。
class RNNModelScratch: #@save
	"""从零开始实现的循环神经⽹络模型"""
	def __init__(self, vocab_size, num_hiddens, device,
					get_params, init_state, forward_fn):
		# 存储从零开始实现的循环神经⽹络模型的参数。
		self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
		self.params = get_params(vocab_size, num_hiddens, device)
		self.init_state, self.forward_fn = init_state, forward_fn
	def __call__(self, X, state):
		X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
		return self.forward_fn(X, state, self.params)
	def begin_state(self, batch_size, device):
		return self.init_state(batch_size, self.num_hiddens, device)

if __name__ == '__main__':
	# 让我们检查输出是否具有正确的形状。例如，隐状态的维数是否保持不变。
	num_hiddens = 512
	net = RNNModelScratch(len(vocab), num_hiddens, train_framework.try_gpu(), get_params,
						init_rnn_state, rnn)
	state = net.begin_state(X.shape[0], train_framework.try_gpu())
	Y, new_state = net(X.to(train_framework.try_gpu()), state)
	# 让我们检查输出是否具有正确的形状。例如，隐状态的维数是否保持不变。
	print("Y.shape, len(new_state), new_state[0].shape : ", 
		Y.shape, len(new_state), new_state[0].shape)

# 让我们⾸先定义预测函数来⽣成prefix之后的新字符。
# 函数包括5个参数，其中：
#    prefix是⼀个⽤⼾提供的包含多个字符的字符串。
#    num_preds是需要生成的后续字符数。
#    net是运算网络。
#    vocab是词元列表。
#    device是设备选择。
def predict_ch8(prefix, num_preds, net, vocab, device): #@save
	"""在prefix后⾯⽣成新字符"""
	state = net.begin_state(batch_size=1, device=device)
	outputs = [vocab[prefix[0]]]
	get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
	# 在循环遍历prefix中的开始字符时，我们不断地将隐状态传递到下⼀个时间步，
	# 但是不⽣成任何输出。这被称为预热（warm-up）期。
	# 因为在此期间模型会⾃我更新（例如，更新隐状态），但不会进⾏预测。
	# 预热期结束后，隐状态的值通常⽐刚开始的初始值更适合预测，从⽽预测字符并输出它们。
	for y in prefix[1:]: # 预热期
		_, state = net(get_input(), state)
		outputs.append(vocab[y])
	for _ in range(num_preds): # 预测num_preds步
		y, state = net(get_input(), state)
		outputs.append(int(y.argmax(dim=1).reshape(1)))
	return ''.join([vocab.idx_to_token[i] for i in outputs])

if __name__ == '__main__':
	# 现在我们可以测试predict_ch8函数。我们将前缀指定为time traveller，
	# 并基于这个前缀⽣成10个后续字符。
	# 鉴于我们还没有训练⽹络，它会⽣成荒谬的预测结果。
	predict_ch8('time traveller ', 10, net, vocab, train_framework.try_gpu())

# 下⾯我们定义⼀个函数来裁剪模型的梯度。
# 梯度裁剪提供了⼀个快速修复梯度爆炸的⽅法，
# 虽然它并不能完全解决问题，但它是众多有效的技术之⼀。
def grad_clipping(net, theta): #@save
	"""裁剪梯度"""
	# 模型是从零开始实现的模型或由⾼级API构建的模型。
	if isinstance(net, nn.Module):
		params = [p for p in net.parameters() if p.requires_grad]
	else:
		params = net.params
	# 我们在此计算了所有模型参数的梯度的范数。
	norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
	if norm > theta:
		for param in params:
			param.grad[:] *= theta / norm

#@save
# 单步训练函数如下，包含6个参数：
#     net             - 模型。
#     train_iter      - 训练集
#     loss            - 损失函数
#     updater         - 更新模型参数的常⽤函数
#     device          - CPU选择号
#     use_random_iter - 使用随机抽样还是顺序抽样。
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
	"""训练⽹络⼀个迭代周期（定义⻅第8章）"""
	state, timer = None, my_timer.Timer()
	# 创建一个累加器，累计训练损失之和,词元数量。
	metric = train_framework.Accumulator(2) # 训练损失之和,词元数量
	for X, Y in train_iter:
		if state is None or use_random_iter:
			# 在第⼀次迭代或使⽤随机抽样时初始化state
			# 当使⽤随机抽样时，因为每个样本都是在⼀个随机位置抽样的，
			# 因此需要为每个迭代周期重新初始化隐状态。
			state = net.begin_state(batch_size=X.shape[0], device=device)
		else:
			# 如果使用的模型是由⾼级API构建的模型。直接调用detach_方法。
			# 将一个Variable从创建它的图中分离，并把它设置成叶子variable。
			if isinstance(net, nn.Module) and not isinstance(state, tuple):
				# state对于nn.GRU是个张量
				state.detach_()
			else:
				# state对于nn.LSTM或对于我们从零开始实现的模型是个张量
				# 否则如果是我们自己从零开始实现的模型或者是一个张量。
				# 需要循环调用detach_方法，一个个分离。
				for s in state:
					s.detach_()
		# 把Y变成一行。
		y = Y.T.reshape(-1)
		X, y = X.to(device), y.to(device)
		# 调用模型进行计算。
		y_hat, state = net(X, state)
		# 计算损失。
		l = loss(y_hat, y.long()).mean()
		# 更新模型参数。
		# 如果是深度学习框架中内置的优化函数。
		if isinstance(updater, torch.optim.Optimizer):
			updater.zero_grad()
			l.backward()
			grad_clipping(net, 1)
			updater.step()
		else:
			l.backward()
			grad_clipping(net, 1) # 因为已经调⽤了mean函数
			updater(batch_size=1)
		metric.add(l * y.numel(), y.numel())
	return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# 循环神经⽹络模型的训练函数既⽀持从零开始实现，也可以使⽤⾼级API来实现。
# 训练函数如下，包含6个参数：
#     net             - 模型。
#     train_iter      - 训练集
#     vocab           - 词元列表
#     lr。            - 学习率
#     num_epochs      - 训练循环次数
#     device          - CPU选择号
#     use_random_iter - 使用随机抽样还是顺序抽样。
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
					use_random_iter=False):
	"""训练模型（定义⻅第8章）"""
	loss = nn.CrossEntropyLoss()
	my_plt.plt.ion()
	animator = train_framework.Animator(xlabel='epoch', ylabel='perplexity',
					legend=['train'], xlim=[10, num_epochs])
	# 初始化更新模型参数的常⽤函数
	if isinstance(net, nn.Module):
		updater = torch.optim.SGD(net.parameters(), lr)
	else:
		updater = lambda batch_size: train_framework.sgd(net.params, lr, batch_size)
	predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
	# 训练和预测
	for epoch in range(num_epochs):
		ppl, speed = train_epoch_ch8(
			net, train_iter, loss, updater, device, use_random_iter)
		if (epoch + 1) % 10 == 0:
			print(predict('time traveller'))
			animator.add(epoch + 1, [ppl])
	my_plt.plt.ioff()
	my_plt.plt.show()
	print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
	print(predict('time traveller'))
	print(predict('traveller'))

if __name__ == '__main__':
	# 现在，我们训练循环神经⽹络模型。因为我们在数据集中只使⽤了10000个词元，所以模型需要更多的迭代周期来更好地收敛。
	num_epochs, lr = 500, 1
	train_ch8(net, train_iter, vocab, lr, num_epochs, train_framework.try_gpu())

	# 最后，让我们检查⼀下使⽤随机抽样⽅法的结果。
	net = RNNModelScratch(len(vocab), num_hiddens, train_framework.try_gpu(), get_params,
				init_rnn_state, rnn)
	train_ch8(net, train_iter, vocab, lr, num_epochs, train_framework.try_gpu(),
				use_random_iter=True)










