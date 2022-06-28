import torch
from torch import nn
import train_framework

# 下⾯，我们从头开始实现⼀个具有张量的批量规范化层。
# 创建⼀个正确的BatchNorm层。这个层将保持适当的参数：
#     拉伸gamma和偏移beta。
# 此外，我们的层将保存均值和⽅差的移动平均值，以便在模型预测期间随后使⽤。
# 因此上这个函数包括七个参数：
#    X - 输入数据
#    gamma, beta - 拉伸gamma和偏移beta。
#    moving_mean, moving_var - 均值和⽅差的移动平均值
#    momentum - 动量参数。
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
	# 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
	if not torch.is_grad_enabled():
		# 如果是在预测模式下，因为这个时候均值和⽅差都已经计算好了。
		# 直接使⽤传⼊的移动平均所得的均值和⽅差。
		X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
	else:
		# 否则如果是训练模式，这个时候，均值和⽅差需要我们计算。
		assert len(X.shape) in (2, 4)
		# 如果X为2维的张量。说明是单个输⼊和单个输出通道。
		# 这里的判断其实有点草率。但是作为例子够用了。
		if len(X.shape) == 2: 
			# 使⽤全连接层的情况，计算特征维上的均值和⽅差
			mean = X.mean(dim=0)
			var = ((X - mean) ** 2).mean(dim=0)
		# 如果X为4维的张量。说明是多个输⼊和多个输出通道。
		else:
			# 使⽤⼆维卷积层的情况，计算通道维上（axis=1）的均值和⽅差。
			# 这⾥我们需要保持X的形状以便后⾯可以做⼴播运算
			mean = X.mean(dim=(0, 2, 3), keepdim=True)
			var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True) 
		# 训练模式下，⽤当前的均值和⽅差做标准化
		X_hat = (X - mean) / torch.sqrt(var + eps)
		# 更新移动平均的均值和⽅差
		moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
		moving_var = momentum * moving_var + (1.0 - momentum) * var
	Y = gamma * X_hat + beta # 缩放和移位
	return Y, moving_mean.data, moving_var.data

# 注意我们实现层的基础设计模式。
# 通常情况下，我们⽤⼀个单独的函数定义其数学原理，⽐如说上面的batch_norm。
# 然后，我们将此功能集成到⼀个⾃定义层中，
# 主要处理数据移动到训练设备（如GPU）、分配和初始化任何必需的变量、
# 跟踪移动平均线（此处为均值和⽅差）等问题。
# 这就是结合了算法分离思想的MVC架构。就是：
#    batch_norm是算法模块，只管逻辑。
#    BatchNorm是存储器，只管数据。
#    图表绘制由其他类负责。
#    训练函数是控制器。
# 代码如下：
class BatchNorm(nn.Module):
	# num_features：完全连接层的输出数量或卷积层的输出通道数。
	# num_dims：2表⽰完全连接层，4表⽰卷积层
	def __init__(self, num_features, num_dims):
		super().__init__()
		if num_dims == 2:
			shape = (1, num_features)
		else:
			shape = (1, num_features, 1, 1) 
		# 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
		self.gamma = nn.Parameter(torch.ones(shape))
		self.beta  = nn.Parameter(torch.zeros(shape))
		# ⾮模型参数的变量初始化为0和1
		self.moving_mean = torch.zeros(shape)
		self.moving_var  = torch.ones(shape)
	def forward(self, X):
		# 如果X不在内存上，将moving_mean和moving_var
		# 复制到X所在显存上
		if self.moving_mean.device != X.device:
			self.moving_mean = self.moving_mean.to(X.device)
			self.moving_var  = self.moving_var.to(X.device)
		# 保存更新过的moving_mean和moving_var
		Y, self.moving_mean, self.moving_var = batch_norm(
		X, self.gamma, self.beta, self.moving_mean,
		self.moving_var, eps=1e-5, momentum=0.9)
		return Y

# 下⾯我们将其应⽤于LeNet模型
# LeNet（LeNet-5）由两个部分组成：
# 	• 卷积编码器：由两个卷积层组成; 
# 	• 全连接层密集块：由三个全连接层组成。
net = nn.Sequential(
		nn.Conv2d(1, 6, kernel_size=5), 
		#   这里新加入了一个批量规范化层。
		#   批量规范化是在卷积层或全连接层之后、相应的激活函数之前应⽤的。
		BatchNorm(6, num_dims=4), 
		nn.Sigmoid(),
		nn.AvgPool2d(kernel_size=2, stride=2),

		nn.Conv2d(6, 16, kernel_size=5), 
		#   这里新加入了一个批量规范化层。
		#   批量规范化是在卷积层或全连接层之后、相应的激活函数之前应⽤的。
		BatchNorm(16, num_dims=4), 
		nn.Sigmoid(),
		nn.AvgPool2d(kernel_size=2, stride=2), 
		nn.Flatten(),

		nn.Linear(16*4*4, 120), 
		#   这里新加入了一个批量规范化层。
		#   批量规范化是在卷积层或全连接层之后、相应的激活函数之前应⽤的。
		BatchNorm(120, num_dims=2), 
		nn.Sigmoid(),
		
		nn.Linear(120, 84), 
		#   这里新加入了一个批量规范化层。
		#   批量规范化是在卷积层或全连接层之后、相应的激活函数之前应⽤的。
		BatchNorm(84, num_dims=2), 
		nn.Sigmoid(),
		nn.Linear(84, 10))

# 我们将在Fashion-MNIST数据集上训练⽹络。这个代码的主要区别在于学习率⼤得多。
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size)
train_framework.train_ch6(net, train_iter, test_iter, num_epochs, lr, train_framework.try_gpu())

# 让我们来看看从第⼀个批量规范化层中学到的拉伸参数gamma和偏移参数beta。
print("net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)) : ",
	net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))

