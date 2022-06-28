import numpy as np
import torch
from torch import nn

import my_timer
import my_plt
import my_download
import train_framework

if __name__ == '__main__':
	# 让我们来看看如何从数据中有效地⽣成⼩批量。
	# 下⾯我们使⽤NASA开发的测试机翼的数据集不同⻜⾏器产⽣的噪声来⽐较这些优化算法。
	#@save
	my_download.DATA_HUB['airfoil'] = \
		(my_download.DATA_URL + 'airfoil_self_noise.dat', 
			'76e5be1548fd8222e5074cf0faae75edff8cf93f')

# 为⽅便起⻅，我们只使⽤前1, 500样本。数据已作预处理：
#@save
def get_data_ch11(batch_size=10, n=1500):
	data = np.genfromtxt(my_download.download('airfoil'),
					dtype=np.float32, delimiter='\t')
	# 我们移除了均值并将⽅差重新缩放到每个坐标为1。
	data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
	data_iter = train_framework.load_array((data[:n, :-1], data[:n, -1]),
				batch_size, is_train=True)
	return data_iter, data.shape[1]-1

# 下面实现⼩批量随机梯度下降算法
def sgd(params, states, hyperparams):
	for p in params:
		# 但是，我们添加了⼀个状态输⼊states并将超参数放在字典hyperparams中。
		p.data.sub_(hyperparams['lr'] * p.grad)
		p.grad.data.zero_()

# 下⾯实现⼀个通⽤的训练函数，
# 然后可以使⽤⼩批量随机梯度下降以及后续⼩节介绍的其他算法来训练模型
# 对应的参数为trainer_fn函数指针。
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
				feature_dim, num_epochs=2):
	# 初始化模型
	w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
			requires_grad=True) 
	b = torch.zeros((1), requires_grad=True)
	# 它初始化了⼀个线性回归模型，
	net, loss = lambda X: train_framework.linreg(X, w, b), train_framework.squared_loss
	# 训练模型
	animator = train_framework.Animator(xlabel='epoch', ylabel='loss',
						xlim=[0, num_epochs], ylim=[0.22, 0.35])
	n, timer = 0, my_timer.Timer()
	# 开始循环。
	for _ in range(num_epochs):
		# 针对每一批数据
		for X, y in data_iter:
			# 计算损失。
			l = loss(net(X), y).mean()
			# 进⾏“反向传播”
			l.backward()
			# 使⽤⼩批量随机梯度下降以及后续⼩节介绍的其他算法来训练模型
			trainer_fn([w, b], states, hyperparams)
			n += X.shape[0]
			# 每200次，输出一个点。
			if n % 200 == 0:
				timer.stop()
				animator.add(n/X.shape[0]/len(data_iter),
					(train_framework.evaluate_loss(net, data_iter, loss),))
				timer.start()
	my_plt.plt.show()
	print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
	return timer.cumsum(), animator.Y[0]


# 批量梯度下降的优化是如何进⾏的。
def train_sgd(lr, batch_size, num_epochs=2):
	data_iter, feature_dim = get_data_ch11(batch_size)
	return train_ch11(
		sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

if __name__ == '__main__':
	# 这可以通过将⼩批量设置为1500（即样本总数）来实现。
	# 因此，模型参数每个迭代轮数只迭代⼀次。
	gd_res = train_sgd(1, 1500, 10)

	# 当批量⼤⼩为1时，优化使⽤的是随机梯度下降。为了简化实现，我们选择了很⼩的学习率。
	sgd_res = train_sgd(0.005, 1)

	# 当批量⼤⼩等于100时，我们使⽤⼩批量随机梯度下降进⾏优化。
	# 每个迭代轮数所需的时间⽐随机梯度下降和批量梯度下降所需的时间短。
	mini1_res = train_sgd(.4, 100)

	# 将批量⼤⼩减少到10，每个迭代轮数的时间都会增加，因为每批⼯作负载的执⾏效率变得更低。
	mini2_res = train_sgd(.05, 10)

	# 现在我们可以⽐较前四个实验的时间与损失
	my_plt.set_figsize([6, 3])
	my_plt.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
		'time (sec)', 'loss', xlim=[1e-2, 10],
		legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
	my_plt.plt.gca().set_xscale('log')

# 下⾯⽤深度学习框架⾃带算法实现⼀个通⽤的训练函数，我们将在本章中其它⼩节使⽤它。
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
	# 初始化模型
	net = nn.Sequential(nn.Linear(5, 1))
	# 初始化权重为正态分布。
	def init_weights(m):
		if type(m) == nn.Linear:
			torch.nn.init.normal_(m.weight, std=0.01)
	net.apply(init_weights)
	# 设定优化器。
	optimizer = trainer_fn(net.parameters(), **hyperparams)
	# 设定损失函数。
	loss = nn.MSELoss(reduction='none')
	# 初始化动画对象。
	animator = train_framework.Animator(xlabel='epoch', ylabel='loss',
			xlim=[0, num_epochs], ylim=[0.22, 0.35])
	# 启动定时器。
	n, timer = 0, my_timer.Timer()
	for _ in range(num_epochs):
		for X, y in data_iter:
			# 初始化梯度。
			optimizer.zero_grad()
			# 调用模型。
			out = net(X)
			# 计算损失。
			y = y.reshape(out.shape)
			l = loss(out, y)
			# 进行反向传播
			l.mean().backward()
			# 调用优化器。
			optimizer.step()
			n += X.shape[0]
			# 每200次，输出一个点。
			if n % 200 == 0:
				timer.stop()
				# MSELoss计算平⽅误差时不带系数1/2
				animator.add(n/X.shape[0]/len(data_iter),
						(train_framework.evaluate_loss(net, data_iter, loss) / 2,))
				timer.start()
	my_plt.plt.show()
	print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')

# 下⾯使⽤这个训练函数，复现之前的实验。
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)




