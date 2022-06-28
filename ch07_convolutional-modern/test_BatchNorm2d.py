import torch
from torch import nn
import train_framework

# LeNet（LeNet-5）由两个部分组成：
# 	• 卷积编码器：由两个卷积层组成; 
net = nn.Sequential(
		nn.Conv2d(1, 6, kernel_size=5), 
		#   这里新加入了一个批量规范化层。
		#   我们直接使⽤深度学习框架中定义的BatchNorm。
		#   其运⾏速度快得多，因为它的代码已编译为C++或CUDA，⽽我们的⾃定义代码由Python实现。
		#   批量规范化是在卷积层或全连接层之后、相应的激活函数之前应⽤的。
		nn.BatchNorm2d(6), 
		nn.Sigmoid(),
		nn.AvgPool2d(kernel_size=2, stride=2),

		nn.Conv2d(6, 16, kernel_size=5), 
		#   这里新加入了一个批量规范化层。
		#   我们直接使⽤深度学习框架中定义的BatchNorm。
		#   其运⾏速度快得多，因为它的代码已编译为C++或CUDA，⽽我们的⾃定义代码由Python实现。
		#   批量规范化是在卷积层或全连接层之后、相应的激活函数之前应⽤的。
		nn.BatchNorm2d(16), 
		nn.Sigmoid(),
		nn.AvgPool2d(kernel_size=2, stride=2), 
		nn.Flatten(),

		nn.Linear(256, 120), 
		#   这里新加入了一个批量规范化层。
		#   我们直接使⽤深度学习框架中定义的BatchNorm。
		#   其运⾏速度快得多，因为它的代码已编译为C++或CUDA，⽽我们的⾃定义代码由Python实现。
		#   批量规范化是在卷积层或全连接层之后、相应的激活函数之前应⽤的。
		nn.BatchNorm1d(120), 
		nn.Sigmoid(),

		nn.Linear(120, 84), 
		#   这里新加入了一个批量规范化层。
		#   我们直接使⽤深度学习框架中定义的BatchNorm。
		#   其运⾏速度快得多，因为它的代码已编译为C++或CUDA，⽽我们的⾃定义代码由Python实现。
		#   批量规范化是在卷积层或全连接层之后、相应的激活函数之前应⽤的。
		nn.BatchNorm1d(84), 
		nn.Sigmoid(),
		nn.Linear(84, 10))

# 我们将在Fashion-MNIST数据集上训练⽹络。
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size)
train_framework.train_ch6(net, train_iter, test_iter, num_epochs, lr, train_framework.try_gpu())
