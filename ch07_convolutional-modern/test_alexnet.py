import torch
from torch import nn
import train_framework

# AlexNet由⼋层组成：五个卷积层、两个全连接隐藏层和⼀个全连接输出层。
# AlexNet使⽤ReLU⽽不是sigmoid作为其激活函数。
net = nn.Sequential(
	# AlexNet由⼋层组成：
	#     五个卷积层、
	# 在AlexNet的第⼀层，卷积窗⼝的形状是11×11。
	# 由于ImageNet中⼤多数图像的宽和⾼⽐MNIST图像的多10倍以上，
	# 因此，需要⼀个更⼤的卷积窗⼝来捕获⽬标。

	# 这⾥，我们使⽤⼀个11*11的更⼤窗⼝来捕捉对象。
	# 同时，步幅为4，以减少输出的⾼度和宽度。
	# 另外，输出通道的数⽬远⼤于LeNet
	nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
	nn.MaxPool2d(kernel_size=3, stride=2),
	# 第⼆层中的卷积窗⼝形状被缩减为5×5。
	# 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
	nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
	# 此外，在第⼀层、第⼆层和第五层卷积层之后，加⼊窗⼝形状为3 × 3、步幅为2的最⼤汇聚层。
	# ⽽且，AlexNet的卷积通道数⽬是LeNet的10倍。
	nn.MaxPool2d(kernel_size=3, stride=2),
	# 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
	# 除了最后的卷积层，输出通道的数量进⼀步增加。
	# 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
	# 后面的三个卷积层的卷积窗⼝形状被缩减为3×3。
	nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
	nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
	nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
	nn.MaxPool2d(kernel_size=3, stride=2),
	nn.Flatten(),
	#     两个全连接隐藏层、
	# 在最后⼀个卷积层后有两个全连接层，分别有4096个输出。
	# 这两个巨⼤的全连接层拥有将近1GB的模型参数。
	# 这⾥，全连接层的输出数量是LeNet中的好⼏倍。使⽤dropout层来减轻过拟合
	nn.Linear(6400, 4096), nn.ReLU(),
	nn.Dropout(p=0.5),
	nn.Linear(4096, 4096), nn.ReLU(),
	nn.Dropout(p=0.5),
	#     ⼀个全连接输出层。
	# 最后是输出层。由于这⾥使⽤Fashion-MNIST，所以⽤类别数为10，⽽⾮论⽂中的1000
	nn.Linear(4096, 10))

# 我们构造⼀个⾼度和宽度都为224的单通道数据，来观察每⼀层输出的形状。
X = torch.randn(1, 1, 224, 224)
for layer in net:
	X=layer(X)
	print(layer.__class__.__name__,'output shape:\t',X.shape)

# 我们在这⾥使⽤的是Fashion-MNIST数据集。
# 将AlexNet直接应⽤于FashionMNIST的⼀个问题是，
# Fashion-MNIST图像的分辨率（28×28像素）低于ImageNet图像。
# 为了解决这个问题，我们将它们增加到224×224
batch_size = 128
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size, resize=224)

# 开始训练AlexNet
lr, num_epochs = 0.01, 10
train_framework.train_ch6(net, train_iter, test_iter, num_epochs, lr, train_framework.try_gpu())











