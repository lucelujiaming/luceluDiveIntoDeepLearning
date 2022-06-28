import torch
from torch import nn
import train_framework
import my_timer

# 经典卷积神经⽹络的基本组成部分是下⾯的这个序列：
# 	1. 带填充以保持分辨率的卷积层；
# 	2. ⾮线性激活函数，如ReLU；
# 	3. 汇聚层，如最⼤汇聚层。
# ⽽⼀个VGG块与之类似，由⼀系列卷积层组成，后⾯再加上⽤于空间下采样的最⼤汇聚层。

def vgg_block(num_convs, in_channels, out_channels):
	layers = []
	for _ in range(num_convs):
		layers.append(nn.Conv2d(in_channels, out_channels,
			kernel_size=3, padding=1))
		layers.append(nn.ReLU())
		in_channels = out_channels
	layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
	return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
print("conv_arch : ", conv_arch)

# 下⾯的代码实现了VGG-11。可以通过在conv_arch上执⾏for循环来简单实现。
def vgg(conv_arch):
	conv_blks = []
	in_channels = 1 # 卷积层部分
	for (num_convs, out_channels) in conv_arch:
		conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
		in_channels = out_channels
	return nn.Sequential(
		*conv_blks, nn.Flatten(),
		# 全连接层部分
		nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
		nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
		nn.Linear(4096, 10))
net = vgg(conv_arch)
print("net : ", net)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
	X = blk(X)
	print(blk.__class__.__name__,'output shape:\t',X.shape)

# 由于VGG-11⽐AlexNet计算量更⼤，因此我们构建了⼀个通道数较少的⽹络。
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size, resize=224)
train_framework.train_ch6(net, train_iter, test_iter, num_epochs, lr, train_framework.try_gpu())


