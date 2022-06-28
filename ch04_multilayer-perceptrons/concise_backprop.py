import torch
import math
import numpy as np
import torch
from torch import nn
from backprop import * 


def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.normal_(m.weight, std=0.01)


if __name__ == '__main__':	
	num_epochs, lr, batch_size = 10, 0.5, 256
	net = nn.Sequential(nn.Flatten(),
	nn.Linear(784, 256),
	nn.ReLU(),
	# 在第⼀个全连接层之后添加⼀个dropout层
	nn.Dropout(dropout1),
	nn.Linear(256, 256),
	nn.ReLU(),
	# 在第⼆个全连接层之后添加⼀个dropout层
	nn.Dropout(dropout2),
	nn.Linear(256, 10))

	loss = nn.CrossEntropyLoss(reduction='none')
	net.apply(init_weights);
	train_iter, test_iter = train_framework.load_data_fashion_mnist(batch_size)
	trainer = torch.optim.SGD(net.parameters(), lr=lr)
	train_framework.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

