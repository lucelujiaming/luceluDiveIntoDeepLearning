import torch
from torch.distributions import multinomial
import my_plt

fair_probs = torch.ones([6]) / 6
print("fair_probs : ", fair_probs)

print("multinomial.Multinomial(1, fair_probs).sample() : ", multinomial.Multinomial(1, fair_probs).sample())
print("multinomial.Multinomial(10, fair_probs).sample() : ", multinomial.Multinomial(10, fair_probs).sample())

counts = multinomial.Multinomial(1000, fair_probs).sample()
print("counts / 1000 : ", counts / 1000) # 相对频率作为估计值

# 让我们进⾏500组实验，每组抽取10个样本。
# 进⾏500组实验
# 作用是对input的每一行做10次取值，使用fair_probs作为权重（概率矩阵）。
# 后面的sample((20,)表明这样试20次。
counts = multinomial.Multinomial(10, fair_probs).sample((500,)) # 20
print("counts : ", counts)
# 返回给定axis上的累计和。累计指的是第n行的值等于前面0 - n行数据的和。
# cumulation summary
# 因为counts每行之和都是10。因此上，cum_counts的每行之和就会为10，20，30....
cum_counts = counts.cumsum(dim=0)
print("cum_counts : ", cum_counts)
# 计算估计值。估计值等于累计和除以总数。
# 明白了cum_counts的含义。我们就会明白，estimates的每一行就是尝试10，20，30...的结果。
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print("cum_counts.sum(dim=1, keepdims=True) : ", cum_counts.sum(dim=1, keepdims=True))
print("estimates : ", estimates)
my_plt.set_figsize((6, 4.5))
for i in range(6):
	my_plt.plt.plot(estimates[:, i].numpy(),
		label=("P(die=" + str(i + 1) + ")"))
my_plt.plt.axhline(y=0.167, color='black', linestyle='dashed')
my_plt.plt.gca().set_xlabel('Groups of experiments')
my_plt.plt.gca().set_ylabel('Estimated probability')
my_plt.plt.legend();
my_plt.plt.show();