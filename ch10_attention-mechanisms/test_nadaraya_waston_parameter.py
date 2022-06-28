import torch
from torch import nn
import my_plt
import test_attention_cues
import train_framework

# Nadaraya-Watson核回归模型是⼀个简单但完整的例⼦，可以⽤于演⽰具有注意⼒机制的机器学习。
n_train = 50 # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5) # 排序后的训练样本

# 根据下⾯的⾮线性函数⽣成⼀个⼈⼯数据集，其中加⼊的噪声项为ϵ：
#     y = 2 * sin(xi) + xi^0.8
def f(x):
	return 2 * torch.sin(x) + x**0.8
# 其中ϵ服从均值为0和标准差为0.5的正态分布。
# 我们⽣成了50个训练样本和50个测试样本。
# 为了更好地可视化之后的注意⼒模式，我们将训练样本进⾏排序。
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,)) # 训练样本的输出
x_test = torch.arange(0, 5, 0.1) # 测试样本
y_truth = f(x_test) # 测试样本的真实输出
n_test = len(x_test) # 测试样本数

X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
torch.bmm(X, Y).shape

weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))

class NWKernelRegression(nn.Module):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
	def forward(self, queries, keys, values):
		# queries和attention_weights的形状为(查询个数，“键－值”对个数)
		queries = queries.repeat_interleave(
					keys.shape[1]).reshape(
							(-1, keys.shape[1]))
		self.attention_weights = nn.functional.softmax(
					-((queries - keys) * self.w)**2 / 2, dim=1)
		# values的形状为(查询个数，“键－值”对个数)
		return torch.bmm(self.attention_weights.unsqueeze(1),
		values.unsqueeze(-1)).reshape(-1)

# X_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输⼊
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = train_framework.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
for epoch in range(5):
	trainer.zero_grad()
	l = loss(net(x_train, keys, values), y_train)
	l.sum().backward()

	trainer.step()
	print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
	animator.add(epoch + 1, float(l.sum()))
my_plt.plt.show()

# 绘制
def plot_kernel_reg(y_hat):
	# 所有的训练样本（样本由圆圈表⽰），
	my_plt.plt.plot(x_train, y_train, 'o', alpha=0.5)
	# my_plt.plt.show()
	# 不带噪声项的真实数据⽣成函数f（标记为“Truth”），
	# 以及学习得到的预测函数（标记为“Pred”）。
	my_plt.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
				xlim=[0, 5], ylim=[-1, 5])
# keys的形状:(n_test，n_train)，每⼀⾏包含着相同的训练输⼊（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

# my_plt.backend_inline.set_matplotlib_formats('png')
test_attention_cues.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
					xlabel='Sorted training inputs',
					ylabel='Sorted testing inputs', usingION = False)
my_plt.plt.ioff()
my_plt.plt.show()

