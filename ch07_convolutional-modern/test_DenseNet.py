import torch
from torch import nn
import train_framework

# DenseNet使⽤了ResNet改良版的“批量规范化、激活和卷积”架构
def conv_block(input_channels, num_channels):
	return nn.Sequential(
		nn.BatchNorm2d(input_channels), nn.ReLU(),
		nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
	def __init__(self, num_convs, input_channels, num_channels):
		super(DenseBlock, self).__init__()
		layer = []
		# ⼀个稠密块由多个卷积块组成，每个卷积块使⽤相同数量的输出通道。
		for i in range(num_convs):
			layer.append(conv_block(
				num_channels * i + input_channels, num_channels))
		self.net = nn.Sequential(*layer)
	def forward(self, X):
		# 然⽽，在前向传播中，我们将每个卷积块的输⼊和输出在通道维上连结。
		for blk in self.net:
			Y = blk(X)
			# 连接通道维度上每个块的输⼊和输出
			X = torch.cat((X, Y), dim=1)
		return X

# 我们定义⼀个有2个输出通道数为10的DenseBlock。使⽤通道数为3的输⼊时，
# 我们会得到通道数为3 + 2 × 10 = 23的输出。
# 卷积块的通道数控制了输出通道数相对于输⼊通道数的增⻓，
# 因此也被称为增⻓率（growth rate）。
blk = DenseBlock(2, 3, 10) 
X = torch.randn(4, 3, 8, 8) 
Y = blk(X)
print("Y.shape : ", Y.shape)

# ⽽过渡层可以⽤来控制模型复杂度。
def transition_block(input_channels, num_channels):
	return nn.Sequential(
		nn.BatchNorm2d(input_channels), 
		nn.ReLU(),
		# 它通过1 × 1卷积层来减⼩通道数，
		nn.Conv2d(input_channels, num_channels, kernel_size=1),
		# 并使⽤步幅为2的平均汇聚层减半⾼和宽，从⽽进⼀步降低模型复杂度。
		nn.AvgPool2d(kernel_size=2, stride=2))

# 对上⼀个例⼦中稠密块的输出使⽤通道数为10的过渡层。
# 此时输出的通道数减为10，⾼和宽均减半。
blk = transition_block(23, 10)
print("blk(Y).shape : ", blk(Y).shape)

# 我们来构造DenseNet模型。
# DenseNet⾸先使⽤同ResNet⼀样的单卷积层和最⼤汇聚层。
b1 = nn.Sequential(
	nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
	nn.BatchNorm2d(64), 
	nn.ReLU(),
	nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# num_channels为当前的通道数
# 稠密块⾥的卷积层通道数（即增⻓率）设为32，所以每个稠密块将增加128个通道。
num_channels, growth_rate = 64, 32
# DenseNet使⽤的是4个稠密块。与ResNet类似，
# 我们可以设置每个稠密块使⽤多少个卷积层。这⾥我们设成4，
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
	blks.append(DenseBlock(num_convs, num_channels, growth_rate))
	# 上⼀个稠密块的输出通道数
	num_channels += num_convs * growth_rate
	# 在稠密块之间添加⼀个转换层，使通道数量减半
	# 在每个模块之间，DenseNet则使⽤过渡层来减半⾼和宽，并减半通道数。
	if i != len(num_convs_in_dense_blocks) - 1:
		blks.append(transition_block(num_channels, num_channels // 2))
		num_channels = num_channels // 2 

# 与ResNet类似，最后接上全局汇聚层和全连接层来输出结果。
net = nn.Sequential(
	b1, 
	*blks,
	nn.BatchNorm2d(num_channels), 
	nn.ReLU(),
	nn.AdaptiveAvgPool2d((1, 1)),
	nn.Flatten(),
	nn.Linear(num_channels, 10))

# 由于这⾥使⽤了⽐较深的⽹络，本节⾥我们将输⼊⾼和宽从224降到96来简化计算。
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size, resize=96)
train_framework.train_ch6(net, train_iter, test_iter, num_epochs, lr, train_framework.try_gpu())




