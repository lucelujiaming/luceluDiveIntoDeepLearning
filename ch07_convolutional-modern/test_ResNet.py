import torch
from torch import nn
from torch.nn import functional as F
import train_framework

# ResNet沿⽤了VGG完整的3 × 3卷积层设计。
class Residual(nn.Module): #@save
	def __init__(self, input_channels, num_channels,
				use_1x1conv=False, strides=1):
		super().__init__()
		# 残差块⾥⾸先有2个有相同输出通道数的3 × 3卷积层。
		self.conv1 = nn.Conv2d(input_channels, num_channels,
						kernel_size=3, padding=1, stride=strides)
		self.conv2 = nn.Conv2d(num_channels, num_channels,
						kernel_size=3, padding=1)
		# 如果想改变通道数，就需要引⼊⼀个额外的1 × 1卷积层
		# 来将输⼊变换成需要的形状后再做相加运算。
		if use_1x1conv:
			self.conv3 = nn.Conv2d(input_channels, num_channels,
						kernel_size=1, stride=strides)
		else:
			self.conv3 = None
		# 上面卷积层需要后接的两个批量规范化层。
		self.bn1 = nn.BatchNorm2d(num_channels)
		self.bn2 = nn.BatchNorm2d(num_channels)
	def forward(self, X):
		# 在执行conv1卷积层后，后接⼀个批量规范化层bn1，后接⼀个ReLU激活函数。
		Y = F.relu(self.bn1(self.conv1(X)))
		# 在执行conv2卷积层后，使用后接⼀个批量规范化层bn2。
		Y = self.bn2(self.conv2(Y))
		# 如果想改变通道数，就需要引⼊⼀个额外的1 × 1卷积层
		# 来将输⼊变换成需要的形状后再做相加运算。
		if self.conv3:
			X = self.conv3(X)
		# 然后我们通过跨层数据通路，跳过上面2个卷积运算，
		# 将输⼊直接加在最后的ReLU激活函数前。
		Y += X
		# 后接⼀个ReLU激活函数。
		return F.relu(Y)

# 我们来查看输⼊和输出形状⼀致的情况。
blk = Residual(3,3) 
X = torch.rand(4, 3, 6, 6) 
Y = blk(X)
print("Y.shape : ", Y.shape)

# 我们也可以在增加输出通道数的同时，减半输出的⾼和宽。
blk = Residual(3,6, use_1x1conv=True, strides=2)
print("blk(X).shape : ", blk(X).shape)

# ResNet的前两层跟之前介绍的GoogLeNet中的⼀样：


b1 = nn.Sequential(
	# 在输出通道数为64、步幅为2的7 × 7卷积层后，
	nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
	# ResNet每个卷积层后增加了批量规范化层。
	nn.BatchNorm2d(64), 
	nn.ReLU(),
	# 之后接步幅为2的3 × 3的最⼤汇聚层。
	nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# ResNet则使⽤4个由残差块组成的模块，每个模块使⽤若⼲个同样输出通道数的残差块。
# 下⾯我们来实现这个模块。注意，我们对第⼀个模块做了特别处理。
def resnet_block(input_channels, num_channels, num_residuals,
					first_block=False):
	blk = []
	for i in range(num_residuals):
		# 第⼀个模块的通道数同输⼊通道数⼀致。
		# 由于之前已经使⽤了步幅为2的最⼤汇聚层，所以⽆须减⼩⾼和宽。
		if i == 0 and not first_block:
			blk.append(Residual(input_channels, num_channels,
						use_1x1conv=True, strides=2))
		# 之后的每个模块在第⼀个残差块⾥将上⼀个模块的通道数翻倍，并将⾼和宽减半。
		else:
			blk.append(Residual(num_channels, num_channels))
	return blk

# 接着在ResNet加⼊所有残差块，这⾥每个模块使⽤2个残差块。
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# 最后，与GoogLeNet⼀样，在ResNet中加⼊全局平均汇聚层，以及全连接层输出。
net = nn.Sequential(b1, b2, b3, b4, b5,
			# 加⼊全局平均汇聚层，
			nn.AdaptiveAvgPool2d((1,1)),
			nn.Flatten(), 
			# 以及全连接层输出。
			nn.Linear(512, 10))

# 让我们观察⼀下ResNet中不同模块的输⼊形状是如何变化的。
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
	X = layer(X)
	print(layer.__class__.__name__,'output shape:\t', X.shape)

# 同之前⼀样，我们在Fashion-MNIST数据集上训练ResNet。
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size, resize=96)
train_framework.train_ch6(net, train_iter, test_iter, num_epochs, lr, train_framework.try_gpu())




