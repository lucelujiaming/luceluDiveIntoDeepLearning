import math
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display
import my_plt
import my_timer

# 下面的函数来自于test_softamx.py。删除了softmax代码后得到。
# 之后加入了之前其他章节写的一些通用函数。不断补充中。。。。

# From test_sgd.py
# 生成测试用的数据：y=Xw+b+噪声。
def synthetic_data(w, b, num_examples): #@save
	"""⽣成y=Xw+b+噪声"""
	# X为正态分布。
	X  = torch.normal(0, 1, (num_examples, len(w)))
	# y=Xw+b
	y  = torch.matmul(X, w) + b 
	# y = y + 噪声。噪声也是正态分布。
	y += torch.normal(0, 0.01, y.shape)
	# y展开成一列。和X返回。
	return X, y.reshape((-1, 1))

def linreg(X, w, b): #@save
	"""线性回归模型"""
	return torch.matmul(X, w) + b

def squared_loss(y_hat, y): #@save
	"""均⽅损失"""
	return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# From test_normal.py
# 实现⼀个函数来评估模型在给定数据集上的损失。
def evaluate_loss(net, data_iter, loss): #@save
	"""评估给定数据集上模型的损失"""
	metric = Accumulator(2) # 损失的总和,样本数量
	for X, y in data_iter:
		# 使用多项式计算X。
		out = net(X)
		# y和X同结构，
		y = y.reshape(out.shape)
		# 计算损失。
		l = loss(out, y)
		metric.add(l.sum(), l.numel())
	return metric[0] / metric[1]

# 2. 调⽤框架中现有的API来读取数据。
def load_array(data_arrays, batch_size, is_train=True): #@save
	"""构造⼀个PyTorch数据迭代器"""
	dataset = data.TensorDataset(*data_arrays)
	return data.DataLoader(dataset, batch_size, shuffle=is_train)

# https://blog.csdn.net/weixin_43256057/article/details/103371367
# 这个文件给出了绘制动画的方法。
# 因为我没有使用Jupyter，因此上需要加入下面的代码：
# 开始训练时加入：
#    my_plt.plt.ion()
# 绘制的时候加入：
#    my_plt.plt.pause(0.01)
# 结束训练的时候加入：
# 	my_plt.plt.ioff()
#	my_plt.plt.show()
# 这样就会生成动画代码。

# Fashion-MNIST中包含的10个类别，分别为：
# t-shirt（T恤）、trouser（裤⼦）、pullover（套衫）、
# dress（连⾐裙）、coat（外套）、sandal（凉鞋）、
# shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
# 以下函数⽤于在数字标签索引及其⽂本名称之间进⾏转换。
def get_fashion_mnist_labels(labels): #@save
	"""返回Fashion-MNIST数据集的⽂本标签"""
	text_labels = ['t-shirt', 'trouser', 'pullover', 
		'dress', 'coat', 'sandal', 'shirt', 
		'sneaker', 'bag', 'ankle boot']

def get_dataloader_workers(): #@save
	"""使⽤4个进程来读取数据"""
	return 4
# 该函数⽤于获取和读取Fashion-MNIST数据集。
# 返回训练集和验证集的数据迭代器。
def load_data_fashion_mnist(batch_size, resize=None): #@save
	"""下载Fashion-MNIST数据集，然后将其加载到内存中"""
	# ToTensor()将shape为(H, W, C)的nump.ndarray或img
	#           转为shape为(C, H, W)的tensor。
	# 之后将每一个数值归一化到[0,1]，
	# 而归一化方法也比较简单，直接除以255即可。
	trans = [transforms.ToTensor()]
	# 如果存在可选参数resize，将图像⼤⼩调整为另⼀种形状。
	# 之后个可选参数resize，⽤来将图像⼤⼩调整为另⼀种形状。
	if resize:
		trans.insert(0, transforms.Resize(resize))
	# 定义转换方式，transforms.Compose将多个转换函数组合起来使用
	trans = transforms.Compose(trans)
	# 通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中。
	mnist_train = torchvision.datasets.FashionMNIST(
		root="../data", train=True, transform=trans, download=True)
	mnist_test = torchvision.datasets.FashionMNIST(
		root="../data", train=False, transform=trans, download=True)
	# 返回训练集和验证集的数据迭代器。
	return (data.DataLoader(mnist_train, batch_size, shuffle=True,
			num_workers=get_dataloader_workers()),
		data.DataLoader(mnist_test, batch_size, shuffle=False,
		num_workers=get_dataloader_workers()))

# 接下来，我们实现 3.4节中引⼊的交叉熵损失函数。
def cross_entropy(y_hat, y):
	return - torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y): #@save
	"""计算预测正确的数量"""
	# 假定第⼆个维度存储每个类的预测分数。
	if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
		# 使⽤argmax获得每⾏中最⼤元素的索引来获得预测类别。
		y_hat = y_hat.argmax(axis=1)
	# 将预测类别与真实y元素进⾏⽐较。
	# 结果是⼀个包含0（错）和1（对）的张量。
	cmp = y_hat.type(y.dtype) == y
	# 最后，我们求和会得到正确预测的数量。
	return float(cmp.type(y.dtype).sum())


# 对于任意数据迭代器data_iter可访问的数据集，我们可以评估在任意模型net的精度。
def evaluate_accuracy(net, data_iter): #@save
	"""计算在指定数据集上模型的精度"""
	if isinstance(net, torch.nn.Module):
		net.eval() # 将模型设置为评估模式
	metric = Accumulator(2) # 正确预测数、预测总数
	with torch.no_grad():
		for X, y in data_iter:
			metric.add(accuracy(net(X), y), y.numel())
	return metric[0] / metric[1]

class Accumulator: #@save
	"""在n个变量上累加"""
	def __init__(self, n):
		self.data = [0.0] * n
	def add(self, *args):
		self.data = [a + float(b) for a, b in zip(self.data, args)]
	def reset(self):
		self.data = [0.0] * len(self.data)
	def __getitem__(self, idx):
		return self.data[idx]

# print("evaluate_accuracy(net, test_iter) : ", 
# 		evaluate_accuracy(net, test_iter))

# 定义⼀个函数来训练⼀个迭代周期。
# 请注意，updater是更新模型参数的常⽤函数，它接受批量⼤⼩作为参数。
# 它可以是sgd函数，也可以是框架的内置优化函数。
def train_epoch_ch3(net, train_iter, loss, updater): #@save
	"""训练模型⼀个迭代周期（定义⻅第3章）"""
	# 将模型设置为训练模式
	if isinstance(net, torch.nn.Module):
		net.train()
	# 训练损失总和、训练准确度总和、样本数
	metric = Accumulator(3)
	for X, y in train_iter:
		# 计算梯度并更新参数
		y_hat = net(X)
		l = loss(y_hat, y)
		if isinstance(updater, torch.optim.Optimizer):
			# 使⽤PyTorch内置的优化器和损失函数
			updater.zero_grad()
			l.mean().backward()
			updater.step()
		else:# 使⽤定制的优化器和损失函数
			l.sum().backward()
			updater(X.shape[0])
		metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
	# 返回训练损失和训练精度
	return metric[0] / metric[2], metric[1] / metric[2]

# 定义⼀个在动画中绘制数据的实⽤程序类Animator
class Animator: #@save
	"""在动画中绘制数据"""
	def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
			ylim=None, xscale='linear', yscale='linear',
			fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
			figsize=(3.5, 2.5)):
		# 增量地绘制多条线
		if legend is None:
			legend = []
		my_plt.use_svg_display()
		self.fig, self.axes = \
			my_plt.plt.subplots(nrows, ncols, figsize=figsize)
		if nrows * ncols == 1:
			self.axes = [self.axes, ]
		# 使⽤lambda函数捕获参数
		self.config_axes = lambda: my_plt.set_axes(
			self.axes[0], xlabel, ylabel, 
			xlim, ylim, xscale, yscale, legend)
		self.X, self.Y, self.fmts = None, None, fmts
	def add(self, x, y):
		# 向图表中添加多个数据点
		if not hasattr(y, "__len__"):
			y = [y]
		n = len(y)
		if not hasattr(x, "__len__"):
			x = [x] * n
		if not self.X:
			self.X = [[] for _ in range(n)]
		if not self.Y:
			self.Y = [[] for _ in range(n)]
		for i, (a, b) in enumerate(zip(x, y)):
			if a is not None and b is not None:
				self.X[i].append(a)
				self.Y[i].append(b)
		self.axes[0].cla()
		for x, y, fmt in zip(self.X, self.Y, self.fmts):
			self.axes[0].plot(x, y, fmt)
		self.config_axes()
		display.display(self.fig)
		display.clear_output(wait=True)
		my_plt.plt.pause(0.01)
		# my_plt.plt.clf()


# 实现训练函数
# 在train_iter访问到的训练数据集上训练⼀个模型net。
# 该训练函数将会运⾏多个迭代周期（由num_epochs指定）。
# 在每个迭代周期结束时，利⽤test_iter访问到的测试数据集对模型进⾏评估。
# 我们将利⽤Animator类来可视化训练进度。
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
	"""训练模型（定义⻅第3章）"""
	my_plt.plt.ion()
	animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
	legend=['train loss', 'train acc', 'test acc'])
	for epoch in range(num_epochs):
		train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
		test_acc = evaluate_accuracy(net, test_iter)
		animator.add(epoch + 1, train_metrics + (test_acc,))
		# my_plt.plt.show()
	train_loss, train_acc = train_metrics
	my_plt.plt.ioff()
	my_plt.plt.show()
#	assert train_loss < 0.5, train_loss
#	assert train_acc <= 1 and train_acc > 0.7, train_acc
#	assert test_acc <= 1 and test_acc > 0.7, test_acc

# 下⾯的函数实现⼩批量随机梯度下降更新。
# 该函数接受模型参数集合、学习速率和批量⼤⼩作为输⼊。
# 每⼀步更新的⼤⼩由学习速率lr决定。
# 传入参数
def sgd(params, lr, batch_size): #@save
	"""⼩批量随机梯度下降"""
	with torch.no_grad():
		for param in params:
			# 因为我们计算的损失是⼀个批量样本的总和，
			# 所以我们⽤批量⼤⼩（batch_size）来规范化步⻓
			param -= lr * param.grad / batch_size
			param.grad.zero_()

# 定义的⼩批量随机梯度下降来优化模型的损失函数，
# 设置学习率为0.1。
lr = 0.1
def updater(batch_size):
	return sgd([W, b], lr, batch_size)

# 现在训练已经完成，我们的模型已经准备好对图像进⾏分类预测。
# 给定⼀系列图像，我们将⽐较它们的实际标签（⽂本输出的第⼀⾏）
# 和模型预测（⽂本输出的第⼆⾏）。
def predict_ch3(net, test_iter, n=6): #@save
	"""预测标签（定义⻅第3章）"""
	for X, y in test_iter:
		break
	trues = get_fashion_mnist_labels(y)
	preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
	print("trues : ", trues)
	print("preds : ", preds)
	titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
	my_plt.show_images(
		X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
	my_plt.plt.show()

# From Chapter 6 -> LeNet
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
	"""使⽤GPU计算模型在数据集上的精度"""
	if isinstance(net, nn.Module):
		net.eval() # 设置为评估模式
		if not device:
			device = next(iter(net.parameters())).device
	# 创建一个累加器，记录正确预测的数量，总预测的数量。
	metric = Accumulator(2)
	with torch.no_grad():
		# 循环数据集。
		for X, y in data_iter:
			if isinstance(X, list):
				# BERT微调所需的（之后将介绍）
				X = [x.to(device) for x in X]
			else:
				# 取出数据特征和数据标签。
				X = X.to(device)
				y = y.to(device)
			# 使用神经网络计算X的感知结果。调用accuracy计算预测正确的数量。
			# 之后和数据集里面的数据标签一起做累加。累加元素个数。
			metric.add(accuracy(net(X), y), y.numel())
	# 返回正确预测的数量和总预测的数量的商。也就是正确预测百分比。
	return metric[0] / metric[1]

#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
	"""⽤GPU训练模型(在第六章定义)"""
	# 使⽤在 4.8.2节中介绍的Xavier随机初始化模型参数。
	def init_weights(m):
		if type(m) == nn.Linear or type(m) == nn.Conv2d:
			nn.init.xavier_uniform_(m.weight)
	# 打开绘图动画开关。
	my_plt.plt.ion()
	# 使⽤在 4.8.2节中介绍的Xavier随机初始化模型参数。
	net.apply(init_weights)
	print('training on', device)
	net.to(device)
	# 使用随机梯度下降法SGD算法作为最优化方法。
	optimizer = torch.optim.SGD(net.parameters(), lr=lr)
	# 使用CrossEntropyLoss作为损失函数。
	loss = nn.CrossEntropyLoss()
	# 初始化一个动画对象。
	animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
					legend=['train loss', 'train acc', 'test acc'])
	# 初始化一个计时器。
	timer, num_batches = my_timer.Timer(), len(train_iter)
	# 开始循环。
	for epoch in range(num_epochs):
		# 创建一个计数器，处理训练损失之和，训练准确率之和，样本数
		metric = Accumulator(3)
		# 调用训练函数。
		net.train()
		for i, (X, y) in enumerate(train_iter):
			# 启动定时器
			timer.start()
			# 初始化优化器梯度。
			optimizer.zero_grad()
			# 从多CPU架构中，获得数据特征和数据标签。
			X, y = X.to(device), y.to(device)
			# 使用神经网络计算X的感知结果。
			y_hat = net(X)
			# 计算损失。
			l = loss(y_hat, y)
			# 根据上面的误差计算结果，进行反向传播。
			l.backward()
			# 更新模型参数。
			optimizer.step()
			# 累计训练损失之和，训练准确率之和，样本数。
			with torch.no_grad():
				metric.add(l * X.shape[0], 
					accuracy(y_hat, y), X.shape[0])
			# 停止定时器。
			timer.stop()
			# 计算训练损失率。
			train_l = metric[0] / metric[2]
			# 计算训练准确率。
			train_acc = metric[1] / metric[2]
			# 绘制曲线。
			# if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
			if (i + 1) % (num_batches // 10) == 0 or i == num_batches - 1:
				animator.add(epoch + (i + 1) / num_batches,
								(train_l, train_acc, None))
		test_acc = evaluate_accuracy_gpu(net, test_iter)
		animator.add(epoch + 1, (None, None, test_acc))
	# 关闭动画开关。
	my_plt.plt.ioff()
	# 阻塞绘制。否则动画绘制完成以后，窗口会自动关闭。
	my_plt.plt.show()
	# 打印执行结果。
	print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, ' 
		f'test acc {test_acc:.3f}')
	print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ' 
		f'on {str(device)}')

#@save
def train_ch6_no_gpu(net, train_iter, test_iter, num_epochs, lr): # , device):
	"""⽤GPU训练模型(在第六章定义)"""
	# 使⽤在 4.8.2节中介绍的Xavier随机初始化模型参数。
	def init_weights(m):
		if type(m) == nn.Linear or type(m) == nn.Conv2d:
			nn.init.xavier_uniform_(m.weight)
	# 打开绘图动画开关。
	my_plt.plt.ion()
	# 使⽤在 4.8.2节中介绍的Xavier随机初始化模型参数。
	net.apply(init_weights)
	# print('training on', device)
	# net.to(device)
	# 使用随机梯度下降法SGD算法作为最优化方法。
	optimizer = torch.optim.SGD(net.parameters(), lr=lr)
	# 使用CrossEntropyLoss作为损失函数。
	loss = nn.CrossEntropyLoss()
	# 初始化一个动画对象。
	animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
					legend=['train loss', 'train acc', 'test acc'])
	# 初始化一个计时器。
	timer, num_batches = my_timer.Timer(), len(train_iter)
	# 开始循环。
	for epoch in range(num_epochs):
		# 创建一个计数器，处理训练损失之和，训练准确率之和，样本数
		metric = Accumulator(3)
		# 调用训练函数。
		net.train()
		for i, (X, y) in enumerate(train_iter):
			# 启动定时器
			timer.start()
			# 初始化优化器梯度。
			optimizer.zero_grad()
			# 从多CPU架构中，获得数据特征和数据标签。
			# X, y = X.to(device), y.to(device)
			# 使用神经网络计算X的感知结果。
			y_hat = net(X)
			# 计算损失。
			l = loss(y_hat, y)
			# 根据上面的误差计算结果，进行反向传播。
			l.backward()
			# 更新模型参数。
			optimizer.step()
			# 累计训练损失之和，训练准确率之和，样本数。
			with torch.no_grad():
				metric.add(l * X.shape[0], 
					accuracy(y_hat, y), X.shape[0])
			# 停止定时器。
			timer.stop()
			# 计算训练损失率。
			train_l = metric[0] / metric[2]
			# 计算训练准确率。
			train_acc = metric[1] / metric[2]
			# 绘制曲线。
			if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
			# if (i + 1) % (num_batches // 10) == 0 or i == num_batches - 1:
				animator.add(epoch + (i + 1) / num_batches,
								(train_l, train_acc, None))
		test_acc = evaluate_accuracy_gpu(net, test_iter)
		animator.add(epoch + 1, (None, None, test_acc))
	# 关闭动画开关。
	my_plt.plt.ioff()
	# 阻塞绘制。否则动画绘制完成以后，窗口会自动关闭。
	my_plt.plt.show()
	# 打印执行结果。
	print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, ' 
		f'test acc {test_acc:.3f}')
	print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ' 
		f'on {str(device)}')

def try_gpu(i=0): #@save
	"""如果存在，则返回gpu(i)，否则返回cpu()"""
	if torch.cuda.device_count() >= i + 1:
		return torch.device(f'cuda:{i}')
	return torch.device('cpu')
def try_all_gpus(): #@save
	"""返回所有可⽤的GPU，如果没有GPU，则返回[cpu(),]"""
	devices = [torch.device(f'cuda:{i}')
		for i in range(torch.cuda.device_count())]
	return devices if devices else [torch.device('cpu')]

# from test_rnn_scratch.py
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
	metric = Accumulator(2) # 训练损失之和,词元数量
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
	animator = Animator(xlabel='epoch', ylabel='perplexity',
					legend=['train'], xlim=[10, num_epochs])
	# 初始化更新模型参数的常⽤函数
	if isinstance(net, nn.Module):
		updater = torch.optim.SGD(net.parameters(), lr)
	else:
		updater = lambda batch_size: sgd(net.params, lr, batch_size)
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

# From test_rnn_concise.py
# 我们为⼀个完整的循环神经⽹络模型定义了⼀个RNNModel类。
# 注意，rnn_layer只包含隐藏的循环层，我们还需要创建⼀个单独的输出层。
#@save
class RNNModel(nn.Module):
	"""循环神经⽹络模型"""
	def __init__(self, rnn_layer, vocab_size, **kwargs):
		super(RNNModel, self).__init__(**kwargs)
		self.rnn = rnn_layer
		self.vocab_size = vocab_size
		self.num_hiddens = self.rnn.hidden_size
		# 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
		if not self.rnn.bidirectional:
			self.num_directions = 1
			self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
		else:
			self.num_directions = 2
			self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
	def forward(self, inputs, state):
		X = F.one_hot(inputs.T.long(), self.vocab_size)
		X = X.to(torch.float32)
		Y, state = self.rnn(X, state)
		# 全连接层⾸先将Y的形状改为(时间步数*批量⼤⼩,隐藏单元数) 
		# 它的输出形状是(时间步数*批量⼤⼩,词表⼤⼩)。
		output = self.linear(Y.reshape((-1, Y.shape[-1])))
		return output, state
	def begin_state(self, device, batch_size=1):
		if not isinstance(self.rnn, nn.LSTM):
			# nn.GRU以张量作为隐状态
			return torch.zeros((self.num_directions * self.rnn.num_layers,
							batch_size, self.num_hiddens),
							device=device)
		else:
			# nn.LSTM以元组作为隐状态
			return (torch.zeros((
				self.num_directions * self.rnn.num_layers,
				batch_size, self.num_hiddens), device=device),
					torch.zeros((
						self.num_directions * self.rnn.num_layers,
						batch_size, self.num_hiddens), device=device))

# From test_seq2seq.py
# 通过零值化屏蔽不相关的项，以便后⾯任何不相关预测的计算都是与零的乘积，结果都等于零。
#@save
def sequence_mask(X, valid_len, value=0):
        """在序列中屏蔽不相关的项"""
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                                        device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

# From test_minibatch_sgd_airfoil.py
# 下⾯实现⼀个通⽤的训练函数，
# 然后可以使⽤⼩批量随机梯度下降以及后续⼩节介绍的其他算法来训练模型
# 对应的参数为trainer_fn函数指针。
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
				feature_dim, num_epochs=2):
	# 初始化模型
	w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
			requires_grad=True) 
	b = torch.zeros((1), requires_grad=True)
	# 它初始化了⼀个线性回归模型，
	net, loss = lambda X: linreg(X, w, b), squared_loss
	# 训练模型
	animator = Animator(xlabel='epoch', ylabel='loss',
						xlim=[0, num_epochs], ylim=[0.22, 0.35])
	n, timer = 0, my_timer.Timer()
	# 开始循环。
	for _ in range(num_epochs):
		# 针对每一批数据
		for X, y in data_iter:
			# 计算损失。
			l = loss(net(X), y).mean()
			# 进⾏“反向传播”
			l.backward()
			# 使⽤⼩批量随机梯度下降以及后续⼩节介绍的其他算法来训练模型
			trainer_fn([w, b], states, hyperparams)
			n += X.shape[0]
			# 每200次，输出一个点。
			if n % 200 == 0:
				timer.stop()
				animator.add(n/X.shape[0]/len(data_iter),
					(evaluate_loss(net, data_iter, loss),))
				timer.start()
	my_plt.plt.show()
	print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
	return timer.cumsum(), animator.Y[0]
# 下⾯⽤深度学习框架⾃带算法实现⼀个通⽤的训练函数，我们将在本章中其它⼩节使⽤它。
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
	# 初始化模型
	net = nn.Sequential(nn.Linear(5, 1))
	# 初始化权重为正态分布。
	def init_weights(m):
		if type(m) == nn.Linear:
			torch.nn.init.normal_(m.weight, std=0.01)
	net.apply(init_weights)
	# 设定优化器。
	optimizer = trainer_fn(net.parameters(), **hyperparams)
	# 设定损失函数。
	loss = nn.MSELoss(reduction='none')
	# 初始化动画对象。
	animator = Animator(xlabel='epoch', ylabel='loss',
			xlim=[0, num_epochs], ylim=[0.22, 0.35])
	# 启动定时器。
	n, timer = 0, my_timer.Timer()
	for _ in range(num_epochs):
		for X, y in data_iter:
			# 初始化梯度。
			optimizer.zero_grad()
			# 调用模型。
			out = net(X)
			# 计算损失。
			y = y.reshape(out.shape)
			l = loss(out, y)
			# 进行反向传播
			l.mean().backward()
			# 调用优化器。
			optimizer.step()
			n += X.shape[0]
			# 每200次，输出一个点。
			if n % 200 == 0:
				timer.stop()
				# MSELoss计算平⽅误差时不带系数1/2
				animator.add(n/X.shape[0]/len(data_iter),
						(evaluate_loss(net, data_iter, loss) / 2,))
				timer.start()
	my_plt.plt.show()
	print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')

# From test_ResNet
# ResNet沿⽤了VGG完整的3 × 3卷积层设计。
class Residual(nn.Module): #@save
	def __init__(self, input_channels, num_channels,
				use_1x1conv=False, strides=1):
		super().__init__()
		# 残差块⾥⾸先有2个有相同输出通道数的3 × 3卷积层。
		self.conv1 = nn.Conv2d(input_channels, num_channels,
						kernel_size=3, padding=1, stride=strides)
		self.conv2 = nn.Conv2d(num_channels, num_channels,
						kernel_size=3, padding=1)
		# 如果想改变通道数，就需要引⼊⼀个额外的1 × 1卷积层
		# 来将输⼊变换成需要的形状后再做相加运算。
		if use_1x1conv:
			self.conv3 = nn.Conv2d(input_channels, num_channels,
						kernel_size=1, stride=strides)
		else:
			self.conv3 = None
		# 上面卷积层需要后接的两个批量规范化层。
		self.bn1 = nn.BatchNorm2d(num_channels)
		self.bn2 = nn.BatchNorm2d(num_channels)
	def forward(self, X):
		# 在执行conv1卷积层后，后接⼀个批量规范化层bn1，后接⼀个ReLU激活函数。
		Y = F.relu(self.bn1(self.conv1(X)))
		# 在执行conv2卷积层后，使用后接⼀个批量规范化层bn2。
		Y = self.bn2(self.conv2(Y))
		# 如果想改变通道数，就需要引⼊⼀个额外的1 × 1卷积层
		# 来将输⼊变换成需要的形状后再做相加运算。
		if self.conv3:
			X = self.conv3(X)
		# 然后我们通过跨层数据通路，跳过上面2个卷积运算，
		# 将输⼊直接加在最后的ReLU激活函数前。
		Y += X
		# 后接⼀个ReLU激活函数。
		return F.relu(Y)

# From test_multiple_gpus_concise
# 我们选择的是ResNet-18，它依然能够容易地和快速地训练。
# 因为输⼊的图像很⼩，所以稍微修改了⼀下。
# 我们在开始时使⽤了更⼩的卷积核、步⻓和填充，⽽且删除了最⼤汇聚层。
#@save
def resnet18(num_classes, in_channels=1):
	"""稍加修改的ResNet-18模型"""
	def resnet_block(in_channels, out_channels, num_residuals,
					first_block=False):
		blk = []
		for i in range(num_residuals):
			if i == 0 and not first_block:
				blk.append(Residual(in_channels, out_channels,
								use_1x1conv=True, strides=2))
			else:
				blk.append(Residual(out_channels, out_channels))
		return nn.Sequential(*blk)

	# 该模型使⽤了更⼩的卷积核、步⻓和填充，⽽且删除了最⼤汇聚层
	net = nn.Sequential(
			nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU())
	net.add_module("resnet_block1", resnet_block(
			64, 64, 2, first_block=True))
	net.add_module("resnet_block2", resnet_block(64, 128, 2))
	net.add_module("resnet_block3", resnet_block(128, 256, 2))
	net.add_module("resnet_block4", resnet_block(256, 512, 2))
	net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
	net.add_module("fc", nn.Sequential(nn.Flatten(),
					nn.Linear(512, num_classes)))
	return net

# From test_CIFAR
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
	"""⽤多GPU进⾏⼩批量训练"""
	if isinstance(X, list):
		# 微调BERT中所需（稍后讨论）
		X = [x.to(devices[0]) for x in X]
	else:
		X = X.to(devices[0])
		y = y.to(devices[0])
	net.train()
	trainer.zero_grad()
	pred = net(X)
	l = loss(pred, y)
	l.sum().backward()
	trainer.step()
	train_loss_sum = l.sum()
	train_acc_sum = accuracy(pred, y)
	return train_loss_sum, train_acc_sum

#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
					devices=try_all_gpus()):
	"""⽤多GPU进⾏模型训练"""
	timer, num_batches = my_timer.Timer(), len(train_iter)
	animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
		legend=['train loss', 'train acc', 'test acc'])
	net = nn.DataParallel(net, device_ids=devices).to(devices[0])
	for epoch in range(num_epochs):
		print("Starting ", epoch, " ..... ")
		# 4个维度：储存训练损失，训练准确度，实例数，特点数
		metric = Accumulator(4)
		for i, (features, labels) in enumerate(train_iter):
			timer.start()
			print("Starting train_batch_ch13 ", i, " in ", epoch, " ..... ")
			l, acc = train_batch_ch13(
				net, features, labels, loss, trainer, devices)
			metric.add(l, acc, labels.shape[0], labels.numel())
			timer.stop()
			if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
				animator.add(epoch + (i + 1) / num_batches,
					(metric[0] / metric[2], metric[1] / metric[3],
					None))
		test_acc = evaluate_accuracy_gpu(net, test_iter)
		animator.add(epoch + 1, (None, None, test_acc))
	print(f'loss {metric[0] / metric[2]:.3f}, train acc ' 
		f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
	print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on ' 
		f'{str(devices)}')
