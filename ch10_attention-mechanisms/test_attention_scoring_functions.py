import math
import torch
from torch import nn

import my_plt
import test_attention_cues
import train_framework

# 为了仅将有意义的词元作为值来获取注意⼒汇聚，我们可以指定⼀个有效序列⻓度（即词元的个数），
# 以便在计算softmax时过滤掉超出指定范围的位置。
# 通过这种⽅式，我们可以在下⾯的masked_softmax函数中实现这样的掩蔽softmax操作。
#@save
def masked_softmax(X, valid_lens):
	"""通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
	# X:3D张量，valid_lens:1D或2D张量，用于进行遮蔽操作的矩阵。
	# 如果没有指定进行遮蔽操作的矩阵，直接调用softmax返回。
	if valid_lens is None:
		return nn.functional.softmax(X, dim=-1)
	else:
		shape = X.shape
		# print("masked_softmax::X.shape : ", shape)
		# 如果指定用于进行遮蔽操作的矩阵为一维。则传入的valid_lens表明是复制次数。
		if valid_lens.dim() == 1:
			# 调用repeat_interleave把valid_lens复制成一个矩阵。
			valid_lens = torch.repeat_interleave(valid_lens, shape[1])
		else:
			# 否则把valid_lens展平。
			valid_lens = valid_lens.reshape(-1) 
		# 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0 
		X = train_framework.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
					value=-1e6)
		# 调用softmax返回。
		return nn.functional.softmax(X.reshape(shape), dim=-1)

if __name__ == '__main__':
	# 考虑由两个2 × 4矩阵表⽰的样本，这两个样本的有效⻓度分别为2和3。
	maskSoftMax = masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
	# 经过掩蔽softmax操作，超出有效⻓度的值都被掩蔽为0。
	print("maskSoftMax : ", maskSoftMax)
	# maskSoftMax :
	#  tensor([[[0.4881, 0.5119, 0.0000, 0.0000],
	#         [0.4542, 0.5458, 0.0000, 0.0000]],
	#
	#        [[0.2827, 0.4817, 0.2357, 0.0000],
	#         [0.2787, 0.2590, 0.4623, 0.0000]]])
	# 我们也可以使⽤⼆维张量，为矩阵样本中的每⼀⾏指定有效⻓度。
	maskSoftMax = masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))
	print("maskSoftMax : ", maskSoftMax)
	# maskSoftMax :
	#   tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
	#          [0.2168, 0.3288, 0.4543, 0.0000]],
	# 
	#         [[0.3199, 0.6801, 0.0000, 0.0000],
	#          [0.2193, 0.2319, 0.3128, 0.2360]]])

# 下⾯我们来实现加性注意⼒。
#@save
class AdditiveAttention(nn.Module):
	"""加性注意⼒"""
	def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
		super(AdditiveAttention, self).__init__(**kwargs)
		self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
		self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
		self.w_v = nn.Linear(num_hiddens, 1, bias=False)
		self.dropout = nn.Dropout(dropout)
	# 参见加性注意⼒（additive attention）的评分函数。
	def forward(self, queries, keys, values, valid_lens):
		queries, keys = self.W_q(queries), self.W_k(keys)
		# 在维度扩展后，
		# queries的形状：(batch_size，查询的个数，1，num_hidden)
		# key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
		# 使⽤⼴播⽅式进⾏求和
		features = queries.unsqueeze(2) + keys.unsqueeze(1)
		features = torch.tanh(features)
		# self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
		# scores的形状：(batch_size，查询的个数，“键-值”对的个数)
		scores = self.w_v(features).squeeze(-1)
		self.attention_weights = masked_softmax(scores, valid_lens)
		# values的形状：(batch_size，“键－值”对的个数，值的维度)
		return torch.bmm(self.dropout(self.attention_weights), values)

if __name__ == '__main__':
	# 我们⽤⼀个⼩例⼦来演⽰上⾯的AdditiveAttention类
	# 其中查询、键和值的形状为（批量⼤⼩，步数或词元序列⻓度，特征⼤⼩），
	# 实际输出为(2, 1, 20)、(2, 10, 2)和(2, 10, 4)。
	# 注意⼒汇聚输出的形状为（批量⼤⼩，查询的步数，值的维度）。
	queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
	# values的⼩批量，两个值矩阵是相同的
	values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
												2, 1, 1)
	valid_lens = torch.tensor([2, 6])
	attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
						dropout=0.1)
	attention.eval()
	print("attention(queries, keys, values, valid_lens) : ", 
		attention(queries, keys, values, valid_lens))

	test_attention_cues.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
						xlabel='Keys', ylabel='Queries', usingION = False)
	my_plt.plt.ioff()
	my_plt.plt.show()

# 缩放点积注意⼒
#@save
class DotProductAttention(nn.Module):
	"""缩放点积注意⼒"""
	def __init__(self, dropout, **kwargs):
		super(DotProductAttention, self).__init__(**kwargs)
		self.dropout = nn.Dropout(dropout)
	# queries的形状：(batch_size，查询的个数，d)
	# keys的形状：(batch_size，“键－值”对的个数，d)
	# values的形状：(batch_size，“键－值”对的个数，值的维度)
	# valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
	def forward(self, queries, keys, values, valid_lens=None):
		d = queries.shape[-1] # 设置transpose_b=True为了交换keys的最后两个维度
		scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
		self.attention_weights = masked_softmax(scores, valid_lens)
		return torch.bmm(self.dropout(self.attention_weights), values)

if __name__ == '__main__':
	queries = torch.normal(0, 1, (2, 1, 2))
	attention = DotProductAttention(dropout=0.5)
	attention.eval()
	print("attention(queries, keys, values, valid_lens) : ", 
		attention(queries, keys, values, valid_lens))

	test_attention_cues.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
		xlabel='Keys', ylabel='Queries', usingION = False)
	my_plt.plt.ioff()
	my_plt.plt.show()





