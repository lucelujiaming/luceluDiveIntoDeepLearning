import torch
import torchvision
from torch import nn

import my_timer
import my_plt
import train_framework

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
	download=True)
my_plt.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
# my_plt.plt.show()

# 为了在预测过程中得到确切的结果，我们通常对训练样本只进⾏图像增⼴，
# 且在预测过程中不使⽤随机操作的图像增⼴。

# 在这⾥，我们只使⽤最简单的随机左右翻转。
# 此外，我们使⽤ToTensor实例将⼀批图像转换为深度学习框架所要求的格式，
train_augs = torchvision.transforms.Compose([
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor()])
test_augs = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor()])

# 我们定义⼀个辅助函数，以便于读取图像和应⽤图像增⼴。
def load_cifar10(is_train, augs, batch_size):
	# PyTorch数据集提供的transform函数应⽤图像增⼴来转化图像。
	dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
								transform=augs, download=True)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
				shuffle=is_train, num_workers=train_framework.get_dataloader_workers())
	return dataloader

# 我们在CIFAR-10数据集上训练 7.6节中的ResNet-18模型
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
	"""⽤多GPU进⾏⼩批量训练"""
	if isinstance(X, list):
		# 微调BERT中所需（稍后讨论）
		X = [x.to(devices[0]) for x in X]
	else:
		X = X.to(devices[0])
		y = y.to(devices[0])
	net.train()
	trainer.zero_grad()
	pred = net(X)
	l = loss(pred, y)
	l.sum().backward()
	trainer.step()
	train_loss_sum = l.sum()
	train_acc_sum = train_framework.accuracy(pred, y)
	return train_loss_sum, train_acc_sum

#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
					devices=train_framework.try_all_gpus()):
	"""⽤多GPU进⾏模型训练"""
	timer, num_batches = my_timer.Timer(), len(train_iter)
	animator = train_framework.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
		legend=['train loss', 'train acc', 'test acc'])
	net = nn.DataParallel(net, device_ids=devices).to(devices[0])
	for epoch in range(num_epochs):
		print("Starting ", epoch, " ..... ")
		# 4个维度：储存训练损失，训练准确度，实例数，特点数
		metric = train_framework.Accumulator(4)
		for i, (features, labels) in enumerate(train_iter):
			timer.start()
			print("Starting train_batch_ch13 ", i, " in ", epoch, " ..... ")
			l, acc = train_batch_ch13(
				net, features, labels, loss, trainer, devices)
			metric.add(l, acc, labels.shape[0], labels.numel())
			timer.stop()
			if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
				animator.add(epoch + (i + 1) / num_batches,
					(metric[0] / metric[2], metric[1] / metric[3],
					None))
		test_acc = train_framework.evaluate_accuracy_gpu(net, test_iter)
		animator.add(epoch + 1, (None, None, test_acc))
	print(f'loss {metric[0] / metric[2]:.3f}, train acc ' 
		f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
	print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on ' 
		f'{str(devices)}')


# 现在，我们可以定义train_with_data_aug函数，使⽤图像增⼴来训练模型。
batch_size, devices, net = 256, train_framework.try_all_gpus(), train_framework.resnet18(10, 3)
def init_weights(m):
	if type(m) in [nn.Linear, nn.Conv2d]:
		nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
	# 该函数获取所有的GPU，将图像增⼴应⽤于训练集，
	train_iter = load_cifar10(True, train_augs, batch_size)
	test_iter = load_cifar10(False, test_augs, batch_size)
	loss = nn.CrossEntropyLoss(reduction="none")
	# 并使⽤Adam作为训练的优化算法，
	trainer = torch.optim.Adam(net.parameters(), lr=lr)
	# 最后调⽤刚刚定义的⽤于训练和评估模型的train_ch13函数。
	train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

# 让我们使⽤基于随机左右翻转的图像增⼴来训练模型。
train_with_data_aug(train_augs, test_augs, net)

