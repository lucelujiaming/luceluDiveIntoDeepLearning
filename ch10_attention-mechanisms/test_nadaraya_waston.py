import torch
from torch import nn
import my_plt
import test_attention_cues

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
print("n_test : ", n_test)

# 绘制
def plot_kernel_reg(y_hat):
	# 所有的训练样本（样本由圆圈表⽰），
	my_plt.plt.plot(x_train, y_train, 'o', alpha=0.5)
	# my_plt.plt.show()
	# 不带噪声项的真实数据⽣成函数f（标记为“Truth”），
	# 以及学习得到的预测函数（标记为“Pred”）。
	my_plt.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
				xlim=[0, 5], ylim=[-1, 5])
# 基于平均汇聚来计算所有训练样本输出值的平均值。
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)

# X_repeat的形状:(n_test,n_train),
# 每⼀⾏都包含着相同的测试输⼊（例如：同样的查询）
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每⼀⾏都包含着要在给定的每个查询的值（y_train）之间分配的注意⼒权重
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意⼒权重
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

test_attention_cues.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
				xlabel='Sorted training inputs',
				ylabel='Sorted testing inputs', usingION = False)
my_plt.plt.ioff()
my_plt.plt.show()

