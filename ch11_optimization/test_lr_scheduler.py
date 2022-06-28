import math
import torch
from torch import nn
from torch.optim import lr_scheduler

import my_plt
import get_data_ch11
import train_framework

# 我们从⼀个简单的问题开始，这个问题可以轻松计算，但⾜以说明要义。
# 为此，我们选择了⼀个稍微现代化的LeNet版本
# （激活函数使⽤relu⽽不是sigmoid，汇聚层使⽤最⼤汇聚层⽽不是平均汇聚层），
def net_fn():
	model = nn.Sequential(
		nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
		nn.MaxPool2d(kernel_size=2, stride=2),
		nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
		nn.MaxPool2d(kernel_size=2, stride=2),
		nn.Flatten(),
		nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
		nn.Linear(120, 84), nn.ReLU(),
		nn.Linear(84, 10))
	return model

loss = nn.CrossEntropyLoss()
device = train_framework.try_gpu()
# 应⽤于Fashion-MNIST数据集
batch_size = 256
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size=batch_size)

# 代码⼏乎与d2l.train_ch6定义在卷积神经⽹络⼀章LeNet⼀节中的相同
# 但是加入了学习率调度器scheduler。默认不起效。
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
				scheduler=None):
	net.to(device)
	animator = train_framework.Animator(xlabel='epoch', xlim=[0, num_epochs],
				legend=['train loss', 'train acc', 'test acc'])
	for epoch in range(num_epochs):
		metric = train_framework.Accumulator(3) # train_loss,train_acc,num_examples
		for i, (X, y) in enumerate(train_iter):
			net.train()
			trainer.zero_grad()
			X, y = X.to(device), y.to(device)
			y_hat = net(X)
			l = loss(y_hat, y)
			l.backward()
			trainer.step()
			with torch.no_grad():
				metric.add(l * X.shape[0], train_framework.accuracy(y_hat, y), X.shape[0])
			train_loss = metric[0] / metric[2]
			train_acc = metric[1] / metric[2]
			if (i + 1) % 50 == 0:
				animator.add(epoch + i / len(train_iter),
							(train_loss, train_acc, None))

		test_acc = train_framework.evaluate_accuracy_gpu(net, test_iter)
		animator.add(epoch+1, (None, None, test_acc))
		# 如果定义了学习率调度器。
		if scheduler:
			# 如果是内置的调度器。
			if scheduler.__module__ == lr_scheduler.__name__:
				# UsingPyTorchIn-Builtscheduler
				scheduler.step()
			# 否则使用用户自定义调度器。
			else:
				# Using custom defined scheduler
				for param_group in trainer.param_groups:
					# 当调⽤更新次数时，它将返回学习率的适当值。
					param_group['lr'] = scheduler(epoch)
	print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, ' 
		f'test acc {test_acc:.3f}')

# 让我们来看看如果使⽤默认设置，调⽤此算法会发⽣什么。
# 例如设学习率为0.3并训练30次迭代。
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
# 留意在超过了某点、测试准确度⽅⾯的进展停滞时，训练准确度将如何继续提⾼。
# 两条曲线之间的间隙表⽰过拟合。
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)

# 我们可以在每个迭代轮数（甚⾄在每个⼩批量）之后向下调整学习率。
# 例如，以动态的⽅式来响应优化的进展情况。
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')

# 更通常⽽⾔，我们应该定义⼀个调度器。
class SquareRootScheduler:
	def __init__(self, lr=0.1):
		self.lr = lr
	# 当调⽤更新次数时，它将返回学习率的适当值。
	def __call__(self, num_update):
		# 让我们定义⼀个简单的⽅法，将学习率设置为η = η0 * (t + 1)^(−1/2) 。
		return self.lr * pow(num_update + 1.0, -0.5)

# 让我们在⼀系列值上绘制它的⾏为。
scheduler = SquareRootScheduler(lr=0.1)
my_plt.plot(torch.arange(num_epochs), 
	[scheduler(t) for t in range(num_epochs)])

# 现在让我们来看看这对在Fashion-MNIST数据集上的训练有何影响。
# 我们只是提供调度器作为训练算法的额外参数。
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
						scheduler)

# 虽然我们不可能涵盖所有类型的学习率调度器，
# 但我们会尝试在下⾯简要概述常⽤的策略：
#        多项式衰减和分段常数表。
# 此外，余弦学习率调度在实践中的⼀些问题上运⾏效果很好。
# 在某些问题上，最好在使⽤较⾼的学习率之前预热优化器。

# 多项式衰减的⼀种替代⽅案是乘法衰减
class FactorScheduler:
	def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
		self.factor = factor
		# 设定ηmin。防止学习率衰减超出合理的下限。
		self.stop_factor_lr = stop_factor_lr
		self.base_lr = base_lr
	def __call__(self, num_update):
		# 多项式衰减的⼀种替代⽅案是乘法衰减，即ηt+1 ← ηt * α其中α ∈ (0, 1)。
		# 为了防⽌学习率衰减超出合理的下限，更新⽅程经常修改为ηt+1 ← max(ηmin, ηt * α)。
		self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
		return self.base_lr
scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
my_plt.plot(torch.arange(50), [scheduler(t) for t in range(50)])

# 训练深度⽹络的常⻅策略之⼀是保持分段稳定的学习率，并且每隔⼀段时间就⼀定程度学习率降低。
net = net_fn()
# 假设每步中的值减半
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# 具体地说，给定⼀组降低学习率的时间，例如s = {5, 10, 20}每当t ∈ s时降低ηt+1 ← ηt * α。
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
def get_lr(trainer, scheduler):
	lr = scheduler.get_last_lr()[0]
	trainer.step()
	scheduler.step()
	return lr
my_plt.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler)
							for t in range(num_epochs)])

# 这种分段恒定学习率调度背后的直觉是，让优化持续进⾏，直到权重向量的分布达到⼀个驻点。
# 此时，我们才将学习率降低，以获得更⾼质量的代理来达到⼀个良好的局部最⼩值。
# 下⾯的例⼦展⽰了如何使⽤这种⽅法产⽣更好的解决⽅案。
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
					scheduler)

# 余弦调度器是 [Loshchilov & Hutter, 2016]提出的⼀种启发式算法。
class CosineScheduler:
	def __init__(self, max_update, base_lr=0.01, final_lr=0,
						warmup_steps=0, warmup_begin_lr=0):
		self.base_lr_orig = base_lr
		self.max_update = max_update
		self.final_lr = final_lr
		self.warmup_steps = warmup_steps
		self.warmup_begin_lr = warmup_begin_lr
		self.max_steps = self.max_update - self.warmup_steps
	# 在此期间学习率将增加⾄初始最⼤值，然后冷却直到优化过程结束。
	def get_warmup_lr(self, epoch):
		# 为了简单起⻅，通常使⽤线性递增。这引出了如下表所⽰的时间表。
		increase = (self.base_lr_orig - self.warmup_begin_lr) \
					* float(epoch) / float(self.warmup_steps)
		return self.warmup_begin_lr + increase
	def __call__(self, epoch):
		# 在某些情况下，初始化参数不⾜以得到良好的解。
		# 对此，⼀⽅⾯，我们可以选择⼀个⾜够⼩的学习率，从⽽防⽌⼀开始发散，
		# 然⽽这样进展太缓慢。另⼀⽅⾯，较⾼的学习率最初就会导致发散。
		# 使⽤预热期，
		if epoch < self.warmup_steps:
			return self.get_warmup_lr(epoch)
		# 它所依据的观点是：我们可能不想在⼀开始就太⼤地降低学习率，
		if epoch <= self.max_update:
			# ⽽且可能希望最终能⽤⾮常⼩的学习率来“改进”解决⽅案。
			# 参见公式(11.11.1)
			self.base_lr = self.final_lr + (
				self.base_lr_orig - self.final_lr) * (1 + math.cos(
				math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
		return self.base_lr
# 在下⾯的⽰例中，我们设置了最⼤更新步数T = 20。
scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
my_plt.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])

# 在计算机视觉中，这个调度可以引出改进的结果。但请注意，如下所⽰，这种改进并不能保证成⽴。
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
			scheduler)

# 使⽤预热期。
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
my_plt.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])

# 注意，观察前5个迭代轮数的性能，⽹络最初收敛得更好。
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
				scheduler)




