import torch
import math
import numpy as np
import torch
from torch import nn
from weight_decay import * 

def train_concise(wd):
	train_framework.my_plt.plt.ion()
	net = nn.Sequential(nn.Linear(num_inputs, 1))
	for param in net.parameters():
		param.data.normal_()
	loss = nn.MSELoss(reduction='none')
	num_epochs, lr = 100, 0.003
	# 偏置参数没有衰减
	trainer = torch.optim.SGD([
		{"params":net[0].weight,'weight_decay': wd},
		{"params":net[0].bias}], lr=lr)
	animator = train_framework.Animator(xlabel='epochs', ylabel='loss', yscale='log',
			xlim=[5, num_epochs], legend=['train', 'test'])
	for epoch in range(num_epochs):
		for X, y in train_iter:
			trainer.zero_grad()
			l = loss(net(X), y)
			l.mean().backward()
			trainer.step()
		if (epoch + 1) % 5 == 0:
			animator.add(epoch + 1,
				(train_framework.evaluate_loss(net, train_iter, loss),
				train_framework.evaluate_loss(net, test_iter, loss)))
	train_framework.my_plt.plt.ioff()
	train_framework.my_plt.plt.show()
	print('w的L2范数：', net[0].weight.norm().item())

if __name__ == '__main__':
	# train_concise(0)
	train_concise(3)


