import math
import time
import numpy as np
import torch
import random
import my_plt

# 生成测试用的数据。
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
# 在这份测试数据中，X就是特征，Y就是标签。
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])

# 该函数接收批量⼤⼩、特征矩阵和标签向量作为输⼊，
# ⽣成⼤⼩为batch_size的⼩批量。每个⼩批量包含⼀组特征和标签。
def data_iter(batch_size, features, labels):
	num_examples = len(features)
	indices = list(range(num_examples))
	# 这些样本是随机读取的，没有特定的顺序
	random.shuffle(indices)
	for i in range(0, num_examples, batch_size):
		batch_indices = torch.tensor(
			indices[i: min(i + batch_size, num_examples)])
		yield features[batch_indices], labels[batch_indices]

# 给w和b赋一个初值。
# 从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，
# 我们的任务是更新这些参数，用来逼近true_w和true_b。
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True) 
# 并将偏置初始化为0。
b = torch.zeros(1, requires_grad=True)
print("w, b : ", w, b)

def linreg(X, w, b): #@save
	"""线性回归模型"""
	return torch.matmul(X, w) + b

def squared_loss(y_hat, y): #@save
	"""均⽅损失"""
	return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

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

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

batch_size = 10
# 循环num_epochs次
for epoch in range(num_epochs):
	# # 使⽤从数据集中随机抽取的⼀个⼩批量，
	for X, y in data_iter(batch_size, features, labels):
		# 然后根据参数计算损失的梯度。
		# 计算方法就是调用linreg函数计算Xw + b。之后计算和y的均⽅差。
		l = loss(net(X, w, b), y) # X和y的⼩批量损失
		# l被重新复制以后，通过反向传播跟踪整个计算图，填充关于每个参数的偏导数。
		l.sum().backward()
		# 接下来，朝着减少损失的⽅向更新我们的参数。
		sgd([w, b], lr, batch_size) # 使⽤参数的梯度更新参数
	# 通过⽐较真实参数和通过训练学到的参数来评估训练的成功程度。
	with torch.no_grad():
		train_l = loss(net(features, w, b), labels)
		print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

