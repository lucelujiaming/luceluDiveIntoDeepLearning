import torch
from torch import nn
from torch.nn import functional as F
import train_framework

# Inception块由四条并⾏路径组成。前三条路径使⽤窗⼝⼤⼩为1 × 1、3 × 3和5 × 5的卷积层，
# 从不同空间⼤⼩中提取信息。中间的两条路径在输⼊上执⾏1 × 1卷积，以减少通道数，从⽽降低模型的复杂
# 性。第四条路径使⽤3 × 3最⼤汇聚层，然后使⽤1 × 1卷积层来改变通道数。这四条路径都使⽤合适的填充
# 来使输⼊与输出的⾼和宽⼀致，最后我们将每条线路的输出在通道维度上连结，并构成Inception块的输出。
# 在Inception块中，通常调整的超参数是每层输出通道数。
class Inception(nn.Module):
	# c1--c4是每条路径的输出通道数
	# 它们可以⽤各种滤波器尺⼨探索图像，这意味着不同⼤⼩的滤波器可以有效地识别不同范围的图像细节。
	# 同时，我们可以为不同的滤波器分配不同数量的参数。
	def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
		super(Inception, self).__init__(**kwargs)
		# Inception块由四条并⾏路径组成。
		# 前三条路径使⽤窗⼝⼤⼩为1 × 1、3 × 3和5 × 5的卷积层，
		# 线路1，单1x1卷积层
		self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1) 
		# 线路2，1x1卷积层后接3x3卷积层
		self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
		self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1) 
		# 线路3，1x1卷积层后接5x5卷积层
		self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
		self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2) 
		# 线路4，3x3最⼤汇聚层后接1x1卷积层
		self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
	def forward(self, x):
		p1 = F.relu(self.p1_1(x))
		p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
		p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
		p4 = F.relu(self.p4_2(self.p4_1(x)))
		# 在通道维度上连结输出
		return torch.cat((p1, p2, p3, p4), dim=1)

# GoogLeNet⼀共使⽤9个Inception块和全局平均汇聚层的堆叠来⽣成其估计值。
# Inception块之间的最⼤汇聚层可降低维度。
# 第⼀个模块类似于AlexNet和LeNet，Inception块的组合从VGG继承，全局平均汇聚层避免了在最后使⽤全连接层。

# 现在，我们逐⼀实现GoogLeNet的每个模块。第⼀个模块使⽤64个通道、7 × 7卷积层。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 第⼆个模块使⽤两个卷积层：第⼀个卷积层是64个通道、1 × 1卷积层；
# 第⼆个卷积层使⽤将通道数量增加三倍的3 × 3卷积层。这对应于Inception块中的第⼆条路径。
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
		nn.ReLU(),
		nn.Conv2d(64, 192, kernel_size=3, padding=1),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第三个模块串联两个完整的Inception块。
b3 = nn.Sequential(
				# 第⼀个Inception块的输出通道数为64 + 128 + 32 + 32 = 256，
				# 四个路径之间的输出通道数量⽐为64 : 128 : 32 : 32 = 2 : 4 : 1 : 1。
				# 第⼆个和第三个路径⾸先将输⼊通道的数量分别减少到96/192 = 1/2和16/192 = 1/12，
				Inception(192, 64, (96, 128), (16, 32), 32),
				# 然后连接第⼆个卷积层。第⼆个Inception块的输出通道数增加到128 + 192 + 96 + 64 = 480，
				# 四个路径之间的输出通道数量⽐为128 : 192 : 96 : 64 = 4 : 6 : 3 : 2。
				# 第⼆条和第三条路径⾸先将输⼊通道的数量分别减少到128/256 = 1/2和32/256 = 1/8。
				Inception(256, 128, (128, 192), (32, 96), 64),
				nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第四模块更加复杂，它串联了5个Inception块，其输出通道数分别是: 
# 这些路径的通道数分配和第三模块中的类似，
# ⾸先是含3×3卷积层的第⼆条路径输出最多通道，其次是仅含1×1卷积层的第⼀条路径，
# 之后是含5×5卷积层的第三条路径和含3×3最⼤汇聚层的第四条路径。
# 其中第⼆、第三条路径都会先按⽐例减⼩通道数。这些⽐例在各个Inception块中都略有不同。
b4 = nn.Sequential(
				# 192 + 208 + 48 + 64 = 512、
				Inception(480, 192, (96, 208), (16, 48), 64),
				# 160 + 224 + 64 + 64 = 512、
				Inception(512, 160, (112, 224), (24, 64), 64),
				# 128 + 256 + 64 + 64 = 512、
				Inception(512, 128, (128, 256), (24, 64), 64),
				# 112 + 288 + 64 + 64 = 528和
				Inception(512, 112, (144, 288), (32, 64), 64),
				# 256 + 320 + 128 + 128 = 832。
				Inception(528, 256, (160, 320), (32, 128), 128),
				nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第五模块包含输出通道数为: 
#   256 + 320 + 128 + 128 = 832和384 + 384 + 128 + 128 = 1024的两个Inception块。
# 其中每条路径通道数的分配思路和第三、第四模块中的⼀致，只是在具体数值上有所不同。
b5 = nn.Sequential(
			Inception(832, 256, (160, 320), (32, 128), 128),
			Inception(832, 384, (192, 384), (48, 128), 128),
			# 需要注意的是，第五模块的后⾯紧跟输出层，
			# 该模块同NiN⼀样使⽤全局平均汇聚层，将每个通道的⾼和宽变成1。
			# 最后我们将输出变成⼆维数组，再接上⼀个输出个数为标签类别数的全连接层。
			nn.AdaptiveAvgPool2d((1,1)),
			nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# 为了使Fashion-MNIST上的训练短⼩精悍，我们将输⼊的⾼和宽从224降到96。
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
	X = layer(X)
	print(layer.__class__.__name__,'output shape:\t', X.shape)

# lr, num_epochs, batch_size = 0.1, 10, 128
lr, num_epochs, batch_size = 0.1, 3, 128
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size, resize=96)
train_framework.train_ch6(net, train_iter, test_iter, num_epochs, lr, train_framework.try_gpu())





