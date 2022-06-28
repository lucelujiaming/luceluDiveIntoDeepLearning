import torch
from torch import nn
import my_timer

# 让我们看看如何通过将Sequential替换为HybridSequential来解决代码中这个瓶颈。
# ⾸先，我们定义⼀个简单的多层感知机。
# ⽣产⽹络的⼯⼚模式
def get_net():
	net = nn.Sequential(nn.Linear(512, 256),
		nn.ReLU(),
		nn.Linear(256, 128),
		nn.ReLU(),
		nn.Linear(128, 2))
	return net

if __name__ == '__main__':
	x = torch.randn(size=(1, 512))
	net = get_net()
	print("Common net(x) : ", net(x))

	# PyTorch是基于命令式编程并且使⽤动态计算图。
	# 为了能够利⽤符号式编程的可移植性和效率，
	# 开发⼈员思考能否将这两种编程模型的优点结合起来，
	# 于是就产⽣了torchscript。
	# torchscript允许⽤⼾使⽤纯命令式编程进⾏开发和调试，
	# 同时能够将⼤多数程序转换为符号式程序，
	# 以便在需要产品级计算性能和部署时使⽤
	net = torch.jit.script(net)
	print("torchscript net(x) : ", net(x))

# 为了证明通过编译获得了性能改进，我们⽐较了混合编程前后执⾏net(x)所需的时间。
# 让我们先定义⼀个度量时间的类，它在本章中在衡量（和改进）模型性能时将⾮常有⽤。
#@save
class Benchmark:
	"""⽤于测量运⾏时间"""
	def __init__(self, description='Done'):
		self.description = description
	def __enter__(self):
		self.timer = my_timer.Timer()
		return self
	def __exit__(self, *args):
		print(f'{self.description}: {self.timer.stop():.4f} sec')

if __name__ == '__main__':
	# 现在我们可以调⽤⽹络两次，⼀次使⽤torchscript，⼀次不使⽤torchscript。
	net = get_net()
	with Benchmark('⽆torchscript'):
		for i in range(1000): net(x)
	net = torch.jit.script(net)
	with Benchmark('有torchscript'):
		for i in range(1000): net(x)

	# 编译模型的好处之⼀是我们可以将模型及其参数序列化（保存）到磁盘。
	net.save('my_mlp')
	# !ls -lh my_mlp*



