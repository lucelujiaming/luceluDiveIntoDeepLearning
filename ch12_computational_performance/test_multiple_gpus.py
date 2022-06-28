import torch
from torch import nn
from torch.nn import functional as F

import train_framework
import my_timer

# 本节将详细介绍如何从零开始并⾏地训练⽹络。
# 下⾯我们将使⽤⼀个简单⽹络来演⽰多GPU训练。
# 我们使⽤ 6.6节中介绍的（稍加修改的）LeNet，从零开始定义它，从⽽详细说明参数交换和同步。
# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]
# 定义模型
def lenet(X, params):
	h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
	h1_activation = F.relu(h1_conv)
	h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
	h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
	h2_activation = F.relu(h2_conv)
	h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
	h2 = h2.reshape(h2.shape[0], -1)
	h3_linear = torch.mm(h2, params[4]) + params[5]
	h3 = F.relu(h3_linear)
	y_hat = torch.mm(h3, params[6]) + params[7]
	return y_hat
# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 对于⾼效的多GPU训练，我们需要两个基本操作。
#   ⾸先，我们需要向多个设备分发参数并附加梯度（get_params）。
#        如果没有参数，就不可能在GPU上评估⽹络。
#   第⼆，需要跨多个设备对参数求和，也就是说，需要⼀个allreduce函数。
def get_params(params, device):
	new_params = [p.to(device) for p in params]
	for p in new_params:
		p.requires_grad_()
	return new_params
# 通过将模型参数复制到⼀个GPU。
new_params = get_params(params, train_framework.try_gpu(0))
print('b1 权重:', new_params[1])
print('b1 梯度:', new_params[1].grad)

# 假设现在有⼀个向量分布在多个GPU上，
# 下⾯的allreduce函数将所有向量相加，并将结果⼴播给所有GPU。
def allreduce(data):
	for i in range(1, len(data)):
		data[0][:] += data[i].to(data[0].device)
	for i in range(1, len(data)):
		data[i][:] = data[0].to(data[i].device)

# 通过在不同设备上创建具有不同值的向量并聚合它们。
data = [torch.ones((1, 2), device=
	train_framework.try_gpu(i)) * (i + 1) for i in range(2)]
print('allreduce之前：\n', data[0], '\n', data[1])
allreduce(data)
print('allreduce之后：\n', data[0], '\n', data[1])

# 我们需要⼀个简单的⼯具函数，将⼀个⼩批量数据均匀地分布在多个GPU上。
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
# AttributeError: module 'torch._C' has no attribute '_scatter'
# split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
# print('output:', split)


# 为了⽅便以后复⽤，我们定义了可以同时拆分数据和标签的split_batch函数。
#@save
def split_batch(X, y, devices):
	"""将X和y拆分到多个设备上"""
	assert X.shape[0] == y.shape[0]
	return (nn.parallel.scatter(X, devices),
			nn.parallel.scatter(y, devices))

# 现在我们可以在⼀个⼩批量上实现多GPU训练。
# 在多个GPU之间同步数据将使⽤刚才讨论的辅助函数allreduce和split_and_load。
def train_batch(X, y, device_params, devices, lr):
	X_shards, y_shards = split_batch(X, y, devices)
	# 在每个GPU上分别计算损失
	ls = [loss(lenet(X_shard, device_W), y_shard).sum()
			for X_shard, y_shard, device_W in zip(
				X_shards, y_shards, device_params)]
	for l in ls: # 反向传播在每个GPU上分别执⾏
		l.backward()
	# 将每个GPU的所有梯度相加，并将其⼴播到所有GPU
	with torch.no_grad():
		for i in range(len(device_params[0])):
			allreduce(
				[device_params[c][i].grad for c in range(len(devices))])
	# 在每个GPU上分别更新模型参数
	for param in device_params:
		train_framework.sgd(param, lr, X.shape[0]) # 在这⾥，我们使⽤全尺⼨的⼩批量

# 现在，我们可以定义训练函数。
def train(num_gpus, batch_size, lr):
	train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size)
	devices = [train_framework.try_gpu(i) for i in range(num_gpus)]
	# 训练函数需要分配GPU并将所有模型参数复制到所有设备。
	# 将模型参数复制到num_gpus个GPU
	device_params = [get_params(params, d) for d in devices]
	num_epochs = 10
	animator = train_framework.Animator('epoch', 'test acc', xlim=[1, num_epochs])
	timer = my_timer.Timer()
	for epoch in range(num_epochs):
		timer.start()
		for X, y in train_iter:
			# 显然，每个⼩批量都是使⽤train_batch函数来处理多个GPU。
			# 为单个⼩批量执⾏多GPU训练
			train_batch(X, y, device_params, devices, lr)
			torch.cuda.synchronize()
		timer.stop()
		# 在GPU0上评估模型
		animator.add(epoch + 1, (train_framework.evaluate_accuracy_gpu(
			lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
	print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，' f'在{str(devices)}')

# 让我们看看在单个GPU上运⾏效果得有多好。⾸先使⽤的批量⼤⼩是256，学习率是0.2。
train(num_gpus=1, batch_size=256, lr=0.2)

# 让我们看看Fashion-MNIST数据集上会发⽣什么。
train(num_gpus=2, batch_size=256, lr=0.2)







