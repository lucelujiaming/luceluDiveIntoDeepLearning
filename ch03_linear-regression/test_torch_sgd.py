import numpy as np
import torch
from torch.utils import data
import my_plt

# 1. 生成测试用的数据。
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

# 这个是用于生成测试数据集的真实的w和b的值。
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 2. 调⽤框架中现有的API来读取数据。
def load_array(data_arrays, batch_size, is_train=True): #@save
	"""构造⼀个PyTorch数据迭代器"""
	dataset = data.TensorDataset(*data_arrays)
	return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print("next(iter(data_iter)) : ", next(iter(data_iter)))

# 3. 定义模型
# nn是神经⽹络的缩写
from torch import nn
# 对于标准深度学习模型，我们可以使⽤框架的预定义好的层。
# 个模型变量net，它是⼀个Sequential类的实例。
# Sequential类将多个层串联在⼀起。
# 当给定输⼊数据时，Sequential实例将数据传⼊到第⼀层，
# 然后将第⼀层的输出作为第⼆层的输⼊，以此类推。
net = nn.Sequential(
	# 我们将两个参数传递到nn.Linear中。第⼀个指定输⼊特征形状，即2，
	# 第⼆个指定输出特征形状，输出特征形状为单个标量，因此为1。
	nn.Linear(2, 1))

# 4. 在使⽤net之前，我们需要初始化模型参数。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 5. 定义损失函数：
# 计算均⽅误差使⽤的是MSELoss类，也称为平⽅L2范数。
# 默认情况下，它返回所有样本损失的平均值。
loss = nn.MSELoss()
# 6. 定义优化算法
# ⼩批量随机梯度下降算法是⼀种优化神经⽹络的标准⼯具。
# ⼩批量随机梯度下降只需要设置lr值，这⾥设置为0.03。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 7. 训练
num_epochs = 3
for epoch in range(num_epochs):
	for X, y in data_iter:
		# 然后根据参数计算损失的梯度。
		l = loss(net(X) ,y)
		trainer.zero_grad()
		# l被重新复制以后，通过反向传播跟踪整个计算图，
		# 填充关于每个参数的偏导数。
		l.backward()
		trainer.step()
	# 计算每个迭代周期后的损失，并打印它来监控训练过程。
	l = loss(net(features), labels)
	print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)



