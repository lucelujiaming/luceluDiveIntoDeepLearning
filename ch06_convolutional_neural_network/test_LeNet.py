import torch
from torch import nn
import train_framework
import my_timer

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
print("gpu : ", try_gpu(), try_gpu(10), try_all_gpus())

# LeNet（LeNet-5）由两个部分组成：
# 	• 卷积编码器：由两个卷积层组成; 
# 	• 全连接层密集块：由三个全连接层组成。
net = nn.Sequential(
		# 每个卷积块中的基本单元是：
		#    ⼀个卷积层、。第⼀卷积层有6个输出通道，卷积核为5 * 5。
		#    使⽤2个像素的填充，来补偿5 × 5卷积核导致的特征减少。
		nn.Conv2d(1, 6, kernel_size=5, padding=2), 
		#    ⼀个sigmoid激活函数
		nn.Sigmoid(),
		#    和⼀个平均汇聚层。
		nn.AvgPool2d(kernel_size=2, stride=2),
		#   ⽽第⼆个卷积层有16个输出通道。第⼆个卷积层没有填充，
		#   因此⾼度和宽度都减少了4个像素。
		nn.Conv2d(6, 16, kernel_size=5), 
		#    ⼀个sigmoid激活函数
		nn.Sigmoid(),
		#   ⼀个平均汇聚层的每个2 × 2池操作通过空间下采样将维数减少4倍。
		nn.AvgPool2d(kernel_size=2, stride=2),
		# 为了将卷积块的输出传递给稠密块，我们必须在⼩批量中展平每个样本。
		nn.Flatten(),
		# LeNet的稠密块有三个全连接层，分别有120、84和10个输出。
		nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
		nn.Linear(120, 84), nn.Sigmoid(),
		nn.Linear(84, 10))

# 们将⼀个⼤⼩为28 × 28的单通道（⿊⽩）图像通过LeNet。
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
	# 通过在每⼀层打印输出的形状，我们可以检查模型。
	X = layer(X)
	print(layer.__class__.__name__,'output shape: \t',X.shape)
# 结果如下：
#    gpu :  cpu cpu [device(type='cpu')]
#    输入为28 × 28的单通道（⿊⽩）图像。
#    通道的数量从输⼊时的1个，增加到第⼀个卷积层之后的6个。
#    Conv2d output shape: 	 torch.Size([1, 6, 28, 28])
#    进行sigmoid激活。
#    Sigmoid output shape: 	 torch.Size([1, 6, 28, 28])
#    池化层。每个汇聚层的⾼度和宽度都减半。
#    AvgPool2d output shape: 	 torch.Size([1, 6, 14, 14])
#    再到第⼆个卷积层之后的16个。
#    Conv2d output shape: 	 torch.Size([1, 16, 10, 10])
#    Sigmoid output shape: 	 torch.Size([1, 16, 10, 10])
#    AvgPool2d output shape: 	 torch.Size([1, 16, 5, 5])
#    16 * 5 * 5 = 400
#    每个全连接层减少维数，最终输出⼀个维数与结果分类数相匹配的输出。
#    Flatten output shape: 	 torch.Size([1, 400])
#    Linear output shape: 	 torch.Size([1, 120])
#    Sigmoid output shape: 	 torch.Size([1, 120])
#    Linear output shape: 	 torch.Size([1, 84])
#    Sigmoid output shape: 	 torch.Size([1, 84])
#    Linear output shape: 	 torch.Size([1, 10])

batch_size = 256
# 我们看看LeNet在Fashion-MNIST数据集上的表现。
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
	"""使⽤GPU计算模型在数据集上的精度"""
	if isinstance(net, nn.Module):
		net.eval() # 设置为评估模式
		if not device:
			device = next(iter(net.parameters())).device
	# 创建一个累加器，记录正确预测的数量，总预测的数量。
	metric = train_framework.Accumulator(2)
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
			metric.add(train_framework.accuracy(net(X), y), y.numel())
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
	train_framework.my_plt.plt.ion()
	# 使⽤在 4.8.2节中介绍的Xavier随机初始化模型参数。
	net.apply(init_weights)
	print('training on', device)
	net.to(device)
	# 使用随机梯度下降法SGD算法作为最优化方法。
	optimizer = torch.optim.SGD(net.parameters(), lr=lr)
	# 使用CrossEntropyLoss作为损失函数。
	loss = nn.CrossEntropyLoss()
	# 初始化一个动画对象。
	animator = train_framework.Animator(xlabel='epoch', xlim=[1, num_epochs],
					legend=['train loss', 'train acc', 'test acc'])
	# 初始化一个计时器。
	timer, num_batches = my_timer.Timer(), len(train_iter)
	# 开始循环。
	for epoch in range(num_epochs):
		# 创建一个计数器，处理训练损失之和，训练准确率之和，样本数
		metric = train_framework.Accumulator(3)
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
					train_framework.accuracy(y_hat, y), X.shape[0])
			# 停止定时器。
			timer.stop()
			# 计算训练损失率。
			train_l = metric[0] / metric[2]
			# 计算训练准确率。
			train_acc = metric[1] / metric[2]
			# 绘制曲线。
			if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
				animator.add(epoch + (i + 1) / num_batches,
								(train_l, train_acc, None))
		test_acc = evaluate_accuracy_gpu(net, test_iter)
		animator.add(epoch + 1, (None, None, test_acc))
	# 关闭动画开关。
	train_framework.my_plt.plt.ioff()
	# 阻塞绘制。否则动画绘制完成以后，窗口会自动关闭。
	train_framework.my_plt.plt.show()
	# 打印执行结果。
	print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, ' 
		f'test acc {test_acc:.3f}')
	print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ' 
		f'on {str(device)}')

if __name__ == '__main__':
	lr, num_epochs = 0.9, 10
	train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())


