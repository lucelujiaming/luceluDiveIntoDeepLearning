import torch
from torch import nn
from torch.nn import functional as F 

x = torch.arange(4)
# save保存变量。
torch.save(x, 'x-file')

x2 = torch.load('x-file')
# 将存储在⽂件中的数据读回内存。
print("x2 :", x2)

y = torch.zeros(4)
# 存储⼀个张量列表，
torch.save([x, y],'x-files')
# 然后把它们读回内存。
x2, y2 = torch.load('x-files')
print("(x2, y2) : ", (x2, y2))

mydict = {'x': x, 'y': y}

# 写⼊或读取从字符串映射到张量的字典。
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print("mydict2 : ", mydict2)

class MLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden = nn.Linear(20, 256)
		self.output = nn.Linear(256, 10)
	def forward(self, x):
		return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
# 将模型的参数存储在⼀个叫做“mlp.params”的⽂件中。
torch.save(net.state_dict(), 'mlp.params')
# 实例化了原始多层感知机模型的⼀个备份。
clone = MLP()
# 直接读取⽂件中存储的参数。
clone.load_state_dict(torch.load('mlp.params'))
print("clone.eval() :", clone.eval())
Y_clone = clone(X)
print("Y_clone == Y :", Y_clone == Y)

