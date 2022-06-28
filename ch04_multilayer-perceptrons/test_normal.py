import math
import numpy as np
import torch
from torch import nn
import test_softmax

# 下面的代码用来生成假数据。
# 生成假数据的公式如下：
#      y = 5 + 1.2 * x / (1!)   + (-3.4) * x^2 / (2!) 
#            + 5.6 * x^3 / (3!) + normal()
max_degree = 20 # 多项式的最⼤阶数
n_train, n_test = 100, 100 # 训练和测试数据集⼤⼩
true_w = np.zeros(max_degree) # 分配⼤量的空间
# 给出生成假数据的系数。
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

# 生成正态分布的随机数。
features = np.random.normal(size=(n_train + n_test, 1))
# 将随机数打乱。
np.random.shuffle(features)
# 生成多项式特征。
# 生成一个max_degree大小的列表[1 ,...,19]。之后求幂。
# np.power([2,3], [3,4])表示分别求 2 的 3 次方和 3 的 4 次方。
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
# gamma函数计算阶乘进行重新缩放。
# 为了避免⾮常⼤的梯度值或损失值。我们将特征从xi调整为xi / i! 的原因，
# 这样可以避免很⼤的i带来的特别⼤的指数值。
for i in range(max_degree):
	poly_features[:, i] /= math.gamma(i + 1) # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
# 把结果乘以参数。计算得到假数据的标签。
labels = np.dot(poly_features, true_w)
# 加上normal分布的噪声。
labels += np.random.normal(scale=0.1, size=labels.shape)

# 这行代码返回：
#    假数据。
#    正态分布的随机数。
#    假数据每一个多项式项的系数。
#    假数据的标签。
# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
	torch.float32) for x in [true_w, features, poly_features, labels]]

print("features[:2], poly_features[:2, :], labels[:2] : \n", 
	features[:2], poly_features[:2, :], labels[:2])

# 实现⼀个函数来评估模型在给定数据集上的损失。
def evaluate_loss(net, data_iter, loss): #@save
	"""评估给定数据集上模型的损失"""
	metric = test_softmax.Accumulator(2) # 损失的总和,样本数量
	for X, y in data_iter:
		# 使用多项式计算X。
		out = net(X)
		# y和X同结构，
		y = y.reshape(out.shape)
		# 计算损失。
		l = loss(out, y)
		metric.add(l.sum(), l.numel())
	return metric[0] / metric[1]

from torch.utils import data
# 2. 调⽤框架中现有的API来读取数据。
def load_array(data_arrays, batch_size, is_train=True): #@save
	"""构造⼀个PyTorch数据迭代器"""
	dataset = data.TensorDataset(*data_arrays)
	return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 定义训练函数。
def train(train_features, test_features, train_labels, test_labels,
			num_epochs=400):
	# 指定损失函数。
	loss = nn.MSELoss(reduction='none')
	# 获得输入数据的形状。
	input_shape = train_features.shape[-1] 
	# 不设置偏置，因为我们已经在多项式中实现了它
	net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
	# 批量大小，不少于10个。
	batch_size = min(10, train_labels.shape[0])
	# 加载数据。
	train_iter = load_array((train_features, train_labels.reshape(-1,1)),
						batch_size)
	test_iter = load_array((test_features, test_labels.reshape(-1,1)),
						batch_size, is_train=False)
	# 指定梯度下降操作。
	trainer = torch.optim.SGD(net.parameters(), lr=0.01)
	# 初始化动画对象。
	animator = test_softmax.Animator(xlabel='epoch', ylabel='loss', yscale='log',
					xlim=[1, num_epochs], ylim=[1e-3, 1e2],
					legend=['train', 'test'])
	# 开始循环。默认是400次。
	for epoch in range(num_epochs):
		test_softmax.train_epoch_ch3(net, train_iter, loss, trainer)
		# 每二十次绘制一次损失。
		if epoch == 0 or (epoch + 1) % 20 == 0:
			animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
					evaluate_loss(net, test_iter, loss)))
	print('weight:', net[0].weight.data.numpy())

# 这里有一个背景就是：数据是用poly_features表示的三阶多项式函数生成的。
# 下面我们也使用poly_features表示的三阶多项式函数。

# 展示三阶多项式函数拟合(正常)
# ⾸先使⽤三阶多项式函数，它与数据⽣成函数的阶数相同。
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
		labels[:n_train], labels[n_train:])

# 下面使用poly_features表示的一阶多项式函数。
# 线性函数拟合(⽋拟合)
# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
		labels[:n_train], labels[n_train:])

# 展示⾼阶多项式函数拟合(过拟合)
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :],
		labels[:n_train], labels[n_train:], num_epochs=1500)





