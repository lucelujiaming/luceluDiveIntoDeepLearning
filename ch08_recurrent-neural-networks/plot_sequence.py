import torch
from torch import nn
import train_framework

T = 1000 # 总共产⽣1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
train_framework.my_plt.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

# 我们仅使⽤前600个“特征－标签”对进⾏训练。
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
	features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本⽤于训练
train_iter = train_framework.load_array(
			(features[:n_train], labels[:n_train]),
			 batch_size, is_train=True)

# 我们使⽤⼀个相当简单的架构训练模型：
#    ⼀个拥有两个全连接层的多层感知机，ReLU激活函数和平⽅损失。
# 初始化⽹络权重的函数
def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight)
# ⼀个简单的多层感知机
def get_net():
	net = nn.Sequential(nn.Linear(4, 10),
				nn.ReLU(),
				nn.Linear(10, 1))
	net.apply(init_weights)
	return net

# 平⽅损失。注意：MSELoss计算平⽅误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
	trainer = torch.optim.Adam(net.parameters(), lr)
	for epoch in range(epochs):
		for X, y in train_iter:
			trainer.zero_grad()
			l = loss(net(X), y)
			l.sum().backward()
			trainer.step()
			print(f'epoch {epoch + 1}, ' 
					f'loss: {train_framework.evaluate_loss(net, train_iter, loss):f}')
net = get_net()
train(net, train_iter, loss, 5, 0.01)

onestep_preds = net(features)
train_framework.my_plt.plot([time, time[tau:]],
		[x.detach().numpy(), onestep_preds.detach().numpy()], 
		'time', 'x', legend=['data', '1-step preds'], xlim=[1, 1000],
		figsize=(6, 3))

multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
	multistep_preds[i] = net(
			multistep_preds[i - tau:i].reshape((1, -1)))
train_framework.my_plt.plot([time, time[tau:], time[n_train + tau:]],
			[x.detach().numpy(), onestep_preds.detach().numpy(),
			multistep_preds[n_train + tau:].detach().numpy()], 'time',
			'x', legend=['data', '1-step preds', 'multistep preds'],
			xlim=[1, 1000], figsize=(6, 3))

max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来⾃x的观测，其时间步从（i+1）到（i+T-tau-max_steps+1）

for i in range(tau):
	features[:, i] = x[i: i + T - tau - max_steps + 1] 
# 列i（i>=tau）是来⾃（i-tau+1）步的预测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
	features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
train_framework.my_plt.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
		[features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
		legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
		figsize=(6, 3))






