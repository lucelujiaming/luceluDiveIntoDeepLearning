import torch
from torch import nn

import test_ResNet
import train_framework
import my_timer

# 我们选择的是ResNet-18，它依然能够容易地和快速地训练。
# 因为输⼊的图像很⼩，所以稍微修改了⼀下。
# 我们在开始时使⽤了更⼩的卷积核、步⻓和填充，⽽且删除了最⼤汇聚层。
#@save
def resnet18(num_classes, in_channels=1):
	"""稍加修改的ResNet-18模型"""
	def resnet_block(in_channels, out_channels, num_residuals,
					first_block=False):
		blk = []
		for i in range(num_residuals):
			if i == 0 and not first_block:
				blk.append(test_ResNet.Residual(in_channels, out_channels,
								use_1x1conv=True, strides=2))
			else:
				blk.append(test_ResNet.Residual(out_channels, out_channels))
		return nn.Sequential(*blk)

	# 该模型使⽤了更⼩的卷积核、步⻓和填充，⽽且删除了最⼤汇聚层
	net = nn.Sequential(
			nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU())
	net.add_module("resnet_block1", resnet_block(
			64, 64, 2, first_block=True))
	net.add_module("resnet_block2", resnet_block(64, 128, 2))
	net.add_module("resnet_block3", resnet_block(128, 256, 2))
	net.add_module("resnet_block4", resnet_block(256, 512, 2))
	net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
	net.add_module("fc", nn.Sequential(nn.Flatten(),
					nn.Linear(512, num_classes)))
	return net

if __name__ == '__main__':
	# 我们将在训练回路中初始化⽹络。请参⻅ 4.8节复习初始化⽅法。
	net = resnet18(10) # 获取GPU列表
	devices = train_framework.try_all_gpus()
# 我们将在训练代码实现中初始化⽹络
def train(net, num_gpus, batch_size, lr):
	# 如前所述，⽤于训练的代码需要执⾏⼏个基本功能才能实现⾼效并⾏：
	train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size)
	devices = [train_framework.try_gpu(i) for i in range(num_gpus)]
	def init_weights(m):
		if type(m) in [nn.Linear, nn.Conv2d]:
			nn.init.normal_(m.weight, std=0.01)
	#    • 需要在所有设备上初始化⽹络参数。
	net.apply(init_weights)
	#    • 在数据集上迭代时，要将⼩批量数据分配到所有设备上。
	# 在多个GPU上设置模型
	net = nn.DataParallel(net, device_ids=devices)
	trainer = torch.optim.SGD(net.parameters(), lr)
	loss = nn.CrossEntropyLoss()
	timer, num_epochs = my_timer.Timer(), 10
	animator = train_framework.Animator('epoch', 'test acc', xlim=[1, num_epochs])
	for epoch in range(num_epochs):
		net.train()
		timer.start()
		for X, y in train_iter:
			trainer.zero_grad()
			X, y = X.to(devices[0]), y.to(devices[0])
			#    • 跨设备并⾏计算损失及其梯度。
			l = loss(net(X), y)
			#    • 聚合梯度，并相应地更新参数。
			l.backward()
			trainer.step()
		timer.stop()
		animator.add(epoch + 1, (train_framework.evaluate_accuracy_gpu(net, test_iter),))
	# 最后，并⾏地计算精确度和发布⽹络的最终性能。
	# 除了需要拆分和聚合数据外，训练代码与前⼏章的实现⾮常相似。
	print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，' 
		f'在{str(devices)}')

if __name__ == '__main__':
	# 接下来我们使⽤2个GPU进⾏训练。
	train(net, num_gpus=2, batch_size=512, lr=0.2)


