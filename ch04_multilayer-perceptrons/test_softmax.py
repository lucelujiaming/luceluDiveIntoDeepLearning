import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display
import my_plt
import my_timer

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


# 定义softmax操作
def softmax(X):
	X_exp = torch.exp(X)
	partition = X_exp.sum(1, keepdim=True)
	return X_exp / partition # 这⾥应⽤了⼴播机制

# 我们实现softmax回归模型。
def netSoftMax(X):
	return softmax(torch.matmul(
		# 使⽤reshape函数将每张原始图像展平为向量。
		X.reshape((-1, W.shape[0])), W) + b)

# 接下来，我们实现 3.4节中引⼊的交叉熵损失函数。
def cross_entropy(y_hat, y):
	return - torch.log(y_hat[range(len(y_hat)), y])

# 得到正确预测的数量。
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
# data_iter的格式为多行，每行包括两个元素。一个是值X，一个是标签y。
def evaluate_accuracy(net, data_iter): #@save
	"""计算在指定数据集上模型的精度"""
	if isinstance(net, torch.nn.Module):
		net.eval() # 将模型设置为评估模式
	# 创建一个累加器，累加正确预测数、预测总数。
	metric = Accumulator(2) # 
	with torch.no_grad():
		for X, y in data_iter:
			# 累加两个元素。分别是正确预测数和预测总数。
			# 前者利用net函数计算得到预测成功的个数，
			# 后者是y里面的元素的个数，也就是总个数。
			metric.add(accuracy(net(X), y), y.numel())
	# 返回正确预测数的累加结果 / 预测总数的累加结果
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
	# 创建一个累加器，累加训练损失总和、训练准确度总和、样本数
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
		# 累加训练损失总和、训练准确度总和、样本数。
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
		# 运⾏一个迭代周期。返回训练损失和训练精度。
		train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
		print("train_metrics : ", train_metrics)
		# train_metrics :  (0.4475098250071208, 0.8477833333333333)
		# 利⽤test_iter访问到的测试数据集对模型进⾏评估。
		# 返回正确预测数的累加结果 / 预测总数的累加结果。
		test_acc = evaluate_accuracy(net, test_iter)
		print("test_acc : ", test_acc)
		# test_acc :  0.8274
		print("train_metrics + (test_acc,) : ", train_metrics + (test_acc,))
		# train_metrics + (test_acc,) :  (0.4475098250071208, 0.8477833333333333, 0.8274)
		# 三个元素分别是'train loss', 'train acc', 'test acc'。
		# 在动画中添加数据点。
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


if __name__ == '__main__':
	batch_size = 256
	train_iter, test_iter = load_data_fashion_mnist(batch_size)
	test_predict_iter = list(test_iter)

	print("train_iter, test_iter : ", train_iter, test_iter)
	# 原始数据集中的每个样本都是28×28的图像。
	# 在本节中，我们将展平每个图像，把它们看作⻓度为784的向量。
	# 因为我们的数据集有10个类别，所以⽹络输出维度为10。
	# 因此，权重将构成⼀个784 × 10的矩阵，偏置将构成⼀个1 × 10的⾏向量。
	num_inputs = 784
	num_outputs = 10
	W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) 
	b = torch.zeros(num_outputs, requires_grad=True)

	# 我们训练模型10个迭代周期。
	# num_epochs = 10
	num_epochs = 3
	train_ch3(netSoftMax, train_iter, test_iter, cross_entropy, num_epochs, updater)

	print("test_iter : ", test_predict_iter)
	predict_ch3(netSoftMax, test_predict_iter)


