import torch
from torch import nn
import test_softmax

batch_size = 256
train_iter, test_iter = test_softmax.load_data_fashion_mnist(batch_size)

# ⾸先，我们将实现⼀个具有单隐藏层的多层感知机，它包含256个隐藏单元。
# 输入大小为784，输出大小为10，隐藏层大小为256。
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 输入层参数。
W1 = nn.Parameter(torch.randn(
num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# 输出层参数。
W2 = nn.Parameter(torch.randn(
num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
# 参数列表
params = [W1, b1, W2, b2]

def relu(X):
	a = torch.zeros_like(X)
	return torch.max(X, a)

# 设定
def net(X):
	X = X.reshape((-1, num_inputs))

	H = relu(X@W1 + b1) # 这⾥“@”代表矩阵乘法
	return (H@W2 + b2)

# 设定损失函数。
loss = nn.CrossEntropyLoss(reduction='none')
# 设定训练循环次数和学习率。
num_epochs, lr = 10, 0.1
# 设定更新操作。
updater = torch.optim.SGD(params, lr=lr)
test_softmax.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

test_softmax.predict_ch3(net, test_iter)

