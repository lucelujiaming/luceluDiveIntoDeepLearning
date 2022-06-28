import torch
from torch import nn

# 该函数接受输⼊张量X和卷积核张量K，并返回输出张量Y。
def corr2d(X, K):  #@save
    """计算二维互相关运算。"""
    # 得到卷积核张量K的大小。
    h, w = K.shape
    # 根据输入张量X和卷积核张量K的大小，创建输出张量Y，初始化为零。
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 进行卷积运算。
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print("corr2d(X, K) : ", corr2d(X, K))

class Conv2D(nn.Module):
	def __init__(self, kernel_size):
		super().__init__()
		# 将weight和bias声明为两个模型参数。
		self.weight = nn.Parameter(torch.rand(kernel_size))
		self.bias = nn.Parameter(torch.zeros(1))
	def forward(self, x):
		# 前向传播函数调⽤corr2d函数并添加偏置。
		return corr2d(x, self.weight) + self.bias

# 如下是卷积层的⼀个简单应⽤：通过找到像素变化的位置，来检测图像中不同颜⾊的边缘。
# 构造⼀个6 × 8像素的⿊⽩图像。
X = torch.ones((6, 8))
X[:, 2:6] = 0
print("⼀个6 × 8像素的⿊⽩图像 : ", X)
# 构造⼀个⾼度为1、宽度为2的卷积核K。
# 输出Y中的1代表从⽩⾊到⿊⾊的边缘，
# -1代表从⿊⾊到⽩⾊的边缘，其他情况的输出为0。
K = torch.tensor([[1.0, -1.0]])
# 我们对参数X（输⼊）和K（卷积核）执⾏互相关运算。
Y = corr2d(X, K)
print("互相关运算 : ", Y)
# 将输⼊的⼆维图像转置，再进⾏如上的互相关运算。
print("corr2d(X.t(), K) : ", corr2d(X.t(), K))

# 学习由X⽣成Y的卷积核。
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 构造⼀个卷积层，将其卷积核初始化为随机张量。
# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
print("⼀个6 × 8像素的⿊⽩图像X : ", X)
Y = Y.reshape((1, 1, 6, 7))
print("之前成功识别出来的边缘Y : ", Y)
lr = 3e-2  # 学习率
# 一开始的梯度为None，权重也是一个随机值。
print("一开始的梯度 : ", conv2d.weight.grad)
print("一开始的权重 : ", conv2d.weight.data.reshape((1, 2)))
# 在每次迭代中，
for i in range(10):
	# 调用我们构建的二维卷积层进行边缘识别。
	Y_hat = conv2d(X)
	# print( "Y_hat[", i, "] : ", Y_hat)
	# 我们⽐较之前成功识别出来的边缘Y与
	# 我们构建的二维卷积层输出的边缘识别之间的平⽅误差，
	l = (Y_hat - Y) ** 2
	# print( "平⽅误差l[", i, "] : ", l)
	# 然后计算梯度来更新卷积核。
	conv2d.zero_grad()
	l.sum().backward()
	# 迭代卷积核
	print("梯度 : ", conv2d.weight.grad)
	conv2d.weight.data[:] -= lr * conv2d.weight.grad
	print("学习结果 : ", conv2d.weight.data.reshape((1, 2)))
	if (i + 1) % 2 == 0:
		print(f'epoch {i+1}, loss {l.sum():.3f}')

print("conv2d.weight.data.reshape((1, 2)) : ", 
	conv2d.weight.data.reshape((1, 2)))

print("Learning result : ", conv2d.weight.data.reshape((1, 2)))




