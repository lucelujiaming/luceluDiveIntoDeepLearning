import torch
from torch import nn

# 实现汇聚层的前向传播。
def pool2d(X, pool_size, mode='max'):
	p_h, p_w = pool_size
	Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			# 输出为输⼊中每个区域的最⼤值或平均值。
			if mode == 'max':
				Y[i, j] = X[i: i + p_h, j: j + p_w].max()
			elif mode == 'avg':
				Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
	return Y

# 验证⼆维最⼤汇聚层的输出。
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print("pool2d(X, (2, 2)) : ", pool2d(X, (2, 2)))
print("pool2d(X, (2, 2), 'avg') : ", pool2d(X, (2, 2), 'avg'))

# 构造了⼀个输⼊张量X，它有四个维度，其中样本数和通道数都是1。
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print("X : ", X)
# 如果我们使⽤形状为(3, 3)的汇聚窗⼝，
# 那么默认情况下，我们得到的步幅形状为(3, 3)。
pool2d = nn.MaxPool2d(3)
print("pool2d(X) : ", pool2d(X))
# 填充和步幅可以⼿动设定。
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print("pool2d(X) : ", pool2d(X))
# 设定⼀个任意⼤⼩的矩形汇聚窗⼝，并分别设定填充和步幅的⾼度和宽度。
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print("pool2d(X) : ", pool2d(X))
# 我们将在通道维度上连结张量X和X + 1，以构建具有2个通道的输⼊。
X = torch.cat((X, X + 1), 1) 
print("X : ", X)
# 汇聚后输出通道的数量仍然是2。
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print("pool2d(X) : ", pool2d(X))
