import torch
from torch import nn

# ⾸先看⼀下具有单隐藏层的多层感知机。
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print("net = ", net)
print("net(X) = ", net(X))
# 这个全连接层包含两个参数，分别是该层的权重和偏置。
# 两者都存储为单精度浮点数（float32）。
print("net[2].state_dict() :", net[2].state_dict())

print("type(net[2].bias) : ", type(net[2].bias))
print("net[2].bias : ", net[2].bias)
print("net[2].bias.data : ", net[2].bias.data)

net[2].weight.grad == None
print("net[2].weight.grad == None = ", net[2].weight.grad == None)

print("net[0].named_parameters : ", 
	*[(name, param.shape) for name, param in net[0].named_parameters()])
print("net.named_parameters : ", 
	*[(name, param.shape) for name, param in net.named_parameters()])

print("net.state_dict()['2.bias'].data : ", net.state_dict()['2.bias'].data)

def block1():
	return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
				nn.Linear(8, 4), nn.ReLU())
# 这个block2包括4个block1。
def block2():
	net = nn.Sequential()
	for i in range(4):
		# 在这⾥嵌套
		net.add_module(f'block {i}', block1())
	return net
# 这个网络包括两个部分。一个是block2，一个是nn.Linear(4, 1)。
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# 结果就是rgnet包括四个block1和一个nn.Linear(4, 1)。
print("rgnet = ", rgnet)
print("rgnet(X) = ", rgnet(X))
# 我们访问第⼀个主要的块中、第⼆个⼦块的第⼀层的偏置项。
print("rgnet[0][1][0].bias.data : ", rgnet[0][1][0].bias.data)

# 将所有权重参数初始化为标准差为0.01的⾼斯随机变量，且将偏置参数设置为0。
def init_normal(m):
	if type(m) == nn.Linear:
		nn.init.normal_(m.weight, mean=0, std=0.01)
		nn.init.zeros_(m.bias)
# 应用到net上。
net.apply(init_normal)
print("net[0].weight.data[0], net[0].bias.data[0] : ", 
	net[0].weight.data[0], net[0].bias.data[0])

# 将所有参数初始化为给定的常数，⽐如初始化为1。
def init_constant(m):
	if type(m) == nn.Linear:
		nn.init.constant_(m.weight, 1)
		nn.init.zeros_(m.bias)
net.apply(init_constant)
print("net[0].weight.data[0], net[0].bias.data[0] : ", 
	net[0].weight.data[0], net[0].bias.data[0])
# 对某些块应⽤不同的初始化⽅法。
# 使⽤Xavier初始化⽅法初始化第⼀个神经⽹络层，
def xavier(m):
	if type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight)
# 将第三个神经⽹络层初始化为常量值42。
def init_42(m):
	if type(m) == nn.Linear:
		nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print("net[0].weight.data[0] : ", net[0].weight.data[0])
print("net[2].weight.data : ", net[2].weight.data)

# ⾃定义初始化
def my_init(m):
	if type(m) == nn.Linear:
		print("Init", *[(name, param.shape)
			for name, param in m.named_parameters()][0])
		# 概率在(-10, 10)之间。
		nn.init.uniform_(m.weight, -10, 10)
		# 概率位于绝对值大于5的区间。
		m.weight.data *= m.weight.data.abs() >= 5
net.apply(my_init)

net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print("net[0].weight.data[0] : ", net[0].weight.data[0])

# 在多个层间共享参数：
#     我们可以定义⼀个稠密层，然后使⽤它的参数来设置另⼀个层的参数。
# 我们需要给共享层⼀个名称，以便可以引⽤它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
			shared, nn.ReLU(),
			shared, nn.ReLU(),
			nn.Linear(8, 1))
print("net(X) = ", net(X))
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同⼀个对象，⽽不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])




