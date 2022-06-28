import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, X):
		# 要从其输⼊中减去均值。
		return X - X.mean()

layer = CenteredLayer()
print("layer(torch.FloatTensor([1, 2, 3, 4, 5])) : ", 
	layer(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print("Y.mean() : ", Y.mean())

# 下⾯我们继续定义具有参数的层。
# 在此实现中，我们使⽤修正线性单元作为激活函数。
class MyLinear(nn.Module):
	# 该层需要输⼊参数：in_units和units，分别表⽰输⼊数和输出数。
	def __init__(self, in_units, units):
		super().__init__()
		# 该层需要两个参数，⼀个⽤于表⽰权重，另⼀个⽤于表⽰偏置项。
		self.weight = nn.Parameter(torch.randn(in_units, units))
		self.bias = nn.Parameter(torch.randn(units,))
	def forward(self, X):
		linear = torch.matmul(X, self.weight.data) + self.bias.data
		return F.relu(linear)

linear = MyLinear(5, 3)
print("linear.weight : ", linear.weight)
# 使⽤⾃定义层直接执⾏前向传播计算。
print("linear(torch.rand(2, 5)) : ", linear(torch.rand(2, 5)))
# 使⽤⾃定义层构建模型，就像使⽤内置的全连接层⼀样使⽤⾃定义层。
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print("net(torch.rand(2, 64)) : ", net(torch.rand(2, 64)))

