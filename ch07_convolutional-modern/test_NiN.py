import torch
from torch import nn
import train_framework
import my_timer

# NiN块
# NiN块以⼀个普通卷积层开始，后⾯是两个1 × 1的卷积层。
# 这两个1 × 1卷积层充当带有ReLU激活函数的逐像素全连接层。
# 第⼀层的卷积窗⼝形状通常由⽤⼾设置。随后的卷积窗⼝形状固定为1 × 1。
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
	return nn.Sequential(
		# 以⼀个普通卷积层开始，第⼀层的卷积窗⼝形状通常由⽤⼾设置。
		nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
		# 这两个1 × 1卷积层充当带有ReLU激活函数的逐像素全连接层。
		nn.ReLU(),
		# 后⾯是两个1 × 1的卷积层。
		# 随后的卷积窗⼝形状固定为1 × 1。
		nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
		nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

net = nn.Sequential(
	# 11 * 11 卷积层(96)，步幅4
	nin_block(1, 96, kernel_size=11, strides=4, padding=0),
	# 3 * 3 最大汇聚层，步幅2
	nn.MaxPool2d(3, stride=2),
	# 5 * 5 卷积层(256)，填充1
	nin_block(96, 256, kernel_size=5, strides=1, padding=2),
	# 3 * 3 最大汇聚层，步幅2
	nn.MaxPool2d(3, stride=2),
	# 3 * 3 卷积层(384)，填充1
	nin_block(256, 384, kernel_size=3, strides=1, padding=1),
	# 3 * 3 最大汇聚层，步幅2
	nn.MaxPool2d(3, stride=2),
	# 加了一个Dropout，随机丢弃一些数据。
	nn.Dropout(0.5),
	# 3 * 3 卷积层(10)，填充1
	# 标签类别数是10
	nin_block(384, 10, kernel_size=3, strides=1, padding=1),
	# 全局平均汇聚层
	nn.AdaptiveAvgPool2d((1, 1)),
	# 将四维的输出转成⼆维的输出，其形状为(批量⼤⼩,10)
	nn.Flatten())

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
	X = layer(X)
	print(layer.__class__.__name__,'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size, resize=224)
# train_framework.train_ch6(net, train_iter, test_iter, num_epochs, lr, train_framework.try_gpu())
train_framework.train_ch6_no_gpu(net, train_iter, test_iter, num_epochs, lr)

