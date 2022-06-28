import math
import torch
from torch import nn

import test_attention_scoring_functions

# 我们可以⽤独⽴学习得到的h组不同的线性投影来变换查询、键和值。
# 然后，这h组变换后的查询、键和值将并⾏地送到注意⼒汇聚中。
# 最后，将这h个注意⼒汇聚的输出拼接在⼀起，
# 并且通过另⼀个可以学习的线性投影进⾏变换，以产⽣最终输出。这种设计被称为多头注意⼒。
#@save
class MultiHeadAttention(nn.Module):
	"""多头注意⼒"""
	# 每个头都可能会关注输⼊的不同部分，可以表⽰⽐简单加权平均值更复杂的函数。
	def __init__(self, key_size, query_size, value_size, num_hiddens,
					num_heads, dropout, bias=False, **kwargs):
		super(MultiHeadAttention, self).__init__(**kwargs)
		self.num_heads = num_heads
		# 选择缩放点积注意⼒作为每⼀个注意⼒头。
		self.attention = test_attention_scoring_functions.DotProductAttention(dropout)
		# 查询的可学习参数。
		self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
		# 键的可学习参数。
		self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
		# 值的可学习参数。
		self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
		# 多头注意⼒输出的可学习参数。
		self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
	def forward(self, queries, keys, values, valid_lens):
		# queries，keys，values的形状:
		# (batch_size，查询或者“键－值”对的个数，num_hiddens)
		# valid_lens 的形状:
		# (batch_size，)或(batch_size，查询的个数) 
		# 经过变换后，输出的queries，keys，values 的形状:
		# (batch_size*num_heads，查询或者“键－值”对的个数，
		# num_hiddens/num_heads)
		# 函数transpose_qkv在下面定义。
		queries = transpose_qkv(self.W_q(queries), self.num_heads)
		keys = transpose_qkv(self.W_k(keys), self.num_heads)
		values = transpose_qkv(self.W_v(values), self.num_heads)
		if valid_lens is not None: # 在轴0，将第⼀项（标量或者⽮量）复制num_heads次，
			# 然后如此复制第⼆项，然后诸如此类。
			valid_lens = torch.repeat_interleave(
					valid_lens, repeats=self.num_heads, dim=0)

		# output的形状:(batch_size*num_heads，查询的个数，
		# num_hiddens/num_heads)
		output = self.attention(queries, keys, values, valid_lens)
		# output_concat的形状:(batch_size，查询的个数，num_hiddens)
		output_concat = transpose_output(output, self.num_heads)
		return self.W_o(output_concat)

#@save
def transpose_qkv(X, num_heads):
	"""为了多注意⼒头的并⾏计算⽽变换形状"""
	# 输⼊X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
	# 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
	# num_hiddens/num_heads)
	X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) 
	# 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
	# num_hiddens/num_heads)
	X = X.permute(0, 2, 1, 3) 
	# 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
	# num_hiddens/num_heads)
	return X.reshape(-1, X.shape[2], X.shape[3])

#@save
def transpose_output(X, num_heads):
	"""逆转transpose_qkv函数的操作"""
	X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
	X = X.permute(0, 2, 1, 3)
	return X.reshape(X.shape[0], X.shape[1], -1)
# 下⾯我们使⽤键和值相同的⼩例⼦来测试我们编写的MultiHeadAttention类。
# 多头注意⼒输出的形状是（batch_size，num_queries，num_hiddens）。
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
					num_hiddens, num_heads, 0.5)
print(attention.eval())

#   MultiHeadAttention(
#   	(attention): DotProductAttention(
#   		(dropout): Dropout(p=0.5, inplace=False))
#   	(W_q): Linear(in_features=100, out_features=100, bias=False)
#   	(W_k): Linear(in_features=100, out_features=100, bias=False)
#   	(W_v): Linear(in_features=100, out_features=100, bias=False)
#   	(W_o): Linear(in_features=100, out_features=100, bias=False) 
#   )


batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print("attention(X, Y, Y, valid_lens).shape : ", 
	attention(X, Y, Y, valid_lens).shape)
