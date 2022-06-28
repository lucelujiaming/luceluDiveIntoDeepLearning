import torch
import math
import numpy as np
import torch
from torch import nn
import train_framework

# ⾸先，我们⽣成⼀些假数据。
# 数据条数。
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# 数据参数。
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# 生成测试用的数据：y=Xw+b+噪声。
train_data = train_framework.synthetic_data(true_w, true_b, n_train)
# 调⽤框架中现有的API来读取测试数据。
train_iter = train_framework.load_array(train_data, batch_size)
# 生成验证用的数据：y=Xw+b+噪声。
test_data = train_framework.synthetic_data(true_w, true_b, n_test)
# 调⽤框架中现有的API来读取验证数据。
test_iter = train_framework.load_array(test_data, batch_size, is_train=False)

# 随机初始化模型参数。
def init_params():
	w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True) 
	b = torch.zeros(1, requires_grad=True)
	return [w, b]

# 定义L2范数惩罚
def l2_penalty(w):
	return torch.sum(w.pow(2)) / 2

def train(lambd):
	w, b = init_params()
	# 采用线性拟合。
	net, loss = lambda X: train_framework.linreg(X, w, b), train_framework.squared_loss
	num_epochs, lr = 100, 0.003
	train_framework.my_plt.plt.ion()
	# 初始化动画对象。
	animator = train_framework.Animator(xlabel='epochs', ylabel='loss', yscale='log',
						xlim=[5, num_epochs], legend=['train', 'test'])
	# 开始训练循环。
	for epoch in range(num_epochs):
		for X, y in train_iter:
			# 增加了L2范数惩罚项，⼴播机制使l2_penalty(w)成为⼀个⻓度为batch_size的向量
			l = loss(net(X), y) + lambd * l2_penalty(w)
			# 计算反向导数。
			l.sum().backward()
			# ⼩批量随机梯度下降更新。
			train_framework.sgd([w, b], lr, batch_size)
		if (epoch + 1) % 5 == 0:
			animator.add(epoch + 1, (train_framework.evaluate_loss(net, train_iter, loss),
				train_framework.evaluate_loss(net, test_iter, loss)))
	train_framework.my_plt.plt.ioff()
	train_framework.my_plt.plt.show()
	print('w的L2范数是：', torch.norm(w).item())


if __name__ == '__main__':
	train(lambd=0)
	train(lambd=3)
