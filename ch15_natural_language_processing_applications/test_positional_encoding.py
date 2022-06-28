import math
import torch
from torch import nn

import test_multihead_attention
import test_attention_cues
import my_plt

if __name__ == '__main__':
	# 由于序列⻓度是n，输⼊和输出的通道数量都是d，
	# 所以卷积层的计算复杂度为: O(knd^2)。而⾃注意⼒具有 O(n2d)计算复杂性。

	# 下⾯的代码⽚段是基于多头注意⼒对⼀个张量完成⾃注意⼒的计算，
	# 张量的形状为（批量⼤⼩，时间步的数⽬或词元序列的⻓度，d）。
	# 输出与输⼊的张量形状相同。
	num_hiddens, num_heads = 100, 5
	attention = test_multihead_attention.MultiHeadAttention(
				num_hiddens, num_hiddens, num_hiddens,
				num_hiddens, num_heads, 0.5)
	print(attention.eval())

	batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
	X = torch.ones((batch_size, num_queries, num_hiddens))
	print(attention(X, X, X, valid_lens).shape)

# ⾃注意⼒则因为并⾏计算⽽放弃了顺序操作。
# 为了使⽤序列的顺序信息，我们通过在输⼊表⽰中添加位置编码来注⼊绝对的或相对的位置信息。
# 我们描述的是基于正弦函数和余弦函数的固定位置编码。
# 乍⼀看，这种基于三⻆函数的设计看起来很奇怪。在解释这个设计之前，
# 让我们先在下⾯的PositionalEncoding类中实现它。
#@save
class PositionalEncoding(nn.Module):
	"""位置编码"""
	def __init__(self, num_hiddens, dropout, max_len=1000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(dropout)
		# 创建⼀个⾜够⻓的P
		self.P = torch.zeros((1, max_len, num_hiddens))
		# 根据公式(10.6.2)计算(i / 10000^(2j/d))
		X = torch.arange(max_len, dtype=torch.float32).reshape(
			-1, 1) / torch.pow(10000, torch.arange(
			0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
		# 根据公式(10.6.2)设定元素。奇数项是cos，偶数项是sin。
		self.P[:, :, 0::2] = torch.sin(X)
		self.P[:, :, 1::2] = torch.cos(X)

	def forward(self, X):
		X = X + self.P[:, :X.shape[1], :].to(X.device)
		return self.dropout(X)

if __name__ == '__main__':
	# 在下⾯的例⼦中，我们可以看到位置嵌⼊矩阵的第6列和第7列的频率⾼于第8列和第9列。
	# 第6列和第7列之间的偏移量（第8列和第9列相同）是由于正弦函数和余弦函数的交替。
	encoding_dim, num_steps = 32, 60
	pos_encoding = PositionalEncoding(encoding_dim, 0)
	pos_encoding.eval()
	X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
	P = pos_encoding.P[:, :X.shape[1], :]
	my_plt.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
			figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])

	for i in range(8):
		print(f'{i}的⼆进制是：{i:>03b}')
	# 在⼆进制表⽰中，较⾼⽐特位的交替频率低于较低⽐特位，与下⾯的热图所⽰相似。
	P = P[0, :, :].unsqueeze(0).unsqueeze(0)
	test_attention_cues.show_heatmaps(P, xlabel='Column (encoding dimension)',
					ylabel='Row (position)', 
					figsize=(3.5, 4), cmap='Blues', usingION = False)
	my_plt.plt.ioff()
	my_plt.plt.show()



