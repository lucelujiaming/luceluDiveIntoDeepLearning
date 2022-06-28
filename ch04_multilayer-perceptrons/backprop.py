import torch
import math
import numpy as np
import torch
from torch import nn
import train_framework

# dropout 丢弃百分比。
def dropout_layer(X, dropout):
	assert 0 <= dropout <= 1 
	# 在本情况中，所有元素都被丢弃
	if dropout == 1:
		return torch.zeros_like(X)
	# 在本情况中，所有元素都被保留
	if dropout == 0:
		return X
	# 生成一个随机数的掩码。大于dropout的为零，小于的为一。
	mask = (torch.rand(X.shape) > dropout).float()
	return mask * X / (1.0 - dropout)

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))

# 使用Fashion-MNIST数据集。我们定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元。
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
# 将第⼀个和第⼆个隐藏层的暂退概率分别设置为0.2和0.5，并且暂退法只在训练期间有效。
dropout1, dropout2 = 0.2, 0.5
class Net(nn.Module):
	def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
				is_training = True):
		super(Net, self).__init__()
		# 输入层维度为28 * 28。
		self.num_inputs = num_inputs
		# 记下来训练模型标志。
		self.training = is_training
		# 具有两个隐藏层。
		self.lin1 = nn.Linear(num_inputs, num_hiddens1)
		self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
		# 最后一个层的维度为输出类别。
		self.lin3 = nn.Linear(num_hiddens2, num_outputs)
		# 使用ReLU归一化。
		self.relu = nn.ReLU()
	def forward(self, X):
		# 执行第一个隐藏层。
		H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
		# 只有在训练模型时才使⽤dropout
		# 在第⼀个全连接层之后添加⼀个dropout层
		if self.training == True:
			# 将第⼀个隐藏层的暂退概率分别设置为0.2
			H1 = dropout_layer(H1, dropout1)
		# 执行第二个隐藏层。
		H2 = self.relu(self.lin2(H1))
		# 只有在训练模型时才使⽤dropout
		# 在第⼆个全连接层之后添加⼀个dropout层
		if self.training == True: 
			# 将第⼆个隐藏层的暂退概率设置为0.5
			H2 = dropout_layer(H2, dropout2)
		# 执行第三个隐藏层。
		out = self.lin3(H2)
		return out

if __name__ == '__main__':
	net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
	
	num_epochs, lr, batch_size = 10, 0.5, 256
	loss = nn.CrossEntropyLoss(reduction='none')
	train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size)
	trainer = torch.optim.SGD(net.parameters(), lr=lr)
	train_framework.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

