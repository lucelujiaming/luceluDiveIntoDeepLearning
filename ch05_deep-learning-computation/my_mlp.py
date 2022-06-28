import torch
from torch import nn
from torch.nn import functional as F

# torch.nn.Sequential是一个Sequential容器，
# 模块将按照构造函数中传递的顺序添加到模块中。另外也可以传入一个有序模块。 
# 下⾯的代码⽣成⼀个⽹络，其中：
net = nn.Sequential(
	# 包含⼀个具有256个单元和ReLU激活函数的全连接隐藏层，
	nn.Linear(20, 256), nn.ReLU(), 
	# 然后是⼀个具有10个隐藏单元且不带激活函数的全连接输出层。
	nn.Linear(256, 10))
X = torch.rand(2, 20)
print("nn.Sequential = ", net)
print("nn.Sequential(X) = ", net(X))

# 因为每⼀个module都继承于nn.Module,都会实现__call__与forward函数。
# 除⾮我们实现⼀个新的运算符，否则我们不必担⼼反向传播函数或参数初始化，系统将⾃动⽣成这些。
class MLP(torch.nn.Module):
	# ⽤模型参数声明层。这⾥，我们声明两个全连接的层
	def __init__(self):
		# 调⽤MLP的⽗类Module的构造函数来执⾏必要的初始化。
		# 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
		# 调⽤⽗类的__init__函数，省去了重复编写模版代码的痛苦。
		super().__init__()
		# 然后，我们实例化两个全连接层，分别为self.hidden和self.out。
		# 它包含⼀个多层感知机，其具有256个隐藏单元的隐藏层和⼀个10维输出层。
		self.hidden = torch.nn.Linear(20, 256) # 隐藏层
		self.out = torch.nn.Linear(256, 10) # 输出层
	# 定义模型的前向传播，即如何根据输⼊X返回所需的模型输出
	def forward(self, X):
		# 注意，这⾥我们使⽤ReLU的函数版本，其在nn.functional模块中定义。
		return self.out(F.relu(self.hidden(X)))
net = MLP()
print("MLP = ", net)
print("MLP(X) = ", net(X))

# 现在我们可以更仔细地看看Sequential类是如何⼯作的。
class MySequential(nn.Module):
	# 我们只需要定义两个关键函数：
	# 1. ⼀种将块逐个追加到列表中的函数。
	def __init__(self, *args):
		super().__init__()
		for idx, module in enumerate(args):
			# 这⾥，module是Module⼦类的⼀个实例。我们把它保存在'Module'类的成员。
			# 变量_modules中。module的类型是OrderedDict
			# 每个Module都有⼀个_modules属性。
			# _modules的主要优点是：在模块的参数初始化过程中，
			# 系统知道在_modules字典中查找需要初始化参数的⼦块。
			self._modules[str(idx)] = module
	# 2. ⼀种前向传播函数，⽤于将输⼊按追加块的顺序传递给块组成的“链条”。
	def forward(self, X):
		# OrderedDict保证了按照成员添加的顺序遍历它们
		for block in self._modules.values():
			X = block(X)
		return X
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print("MySequential = ", net)
print("MySequential(X) = ", net(X))

class FixedHiddenMLP(nn.Module):
	def __init__(self):
		super().__init__()
		# 实现了⼀个隐藏层，其权重（self.rand_weight）在实例化时被随机初始化，之后为常量。
		# 也就是，我们不计算梯度的随机权重参数。因此其在训练期间保持不变。
		# 因此它永远不会被反向传播更新。
		self.rand_weight = torch.rand((20, 20), requires_grad=False)
		# 然后，神经⽹络将这个固定层的输出通过⼀个全连接层。
		self.linear = nn.Linear(20, 20)
	def forward(self, X):
		X = self.linear(X)
		# 使⽤创建的常量参数以及relu和mm函数
		X = F.relu(torch.mm(X, self.rand_weight) + 1) # 复⽤全连接层。这相当于两个全连接层共享参数
		X = self.linear(X)
		# 控制流。
		# 在返回输出之前，模型做了⼀些不寻常的事情：
		# 它运⾏了⼀个while循环，在L1范数⼤于1的条件下，
		# 将输出向量除以2，直到它满⾜条件为⽌。
		while X.abs().sum() > 1:
			X /= 2
		# 最后，模型返回了X中所有项的和。
		return X.sum()
net = FixedHiddenMLP()
print("FixedHiddenMLP = ", net)
print("FixedHiddenMLP(X) = ", net(X))

# 我们可以混合搭配各种组合块的⽅法。
class NestMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
							nn.Linear(64, 32), nn.ReLU())
		self.linear = nn.Linear(32, 16)
	def forward(self, X):
		return self.linear(self.net(X))
net = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print("Sequential(NestMLP, Linear, NestMLP) = ", net)
print("Sequential(NestMLP(X), Linear, NestMLP(X)) = ", net(X))




