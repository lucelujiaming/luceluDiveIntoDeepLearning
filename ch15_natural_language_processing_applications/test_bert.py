import torch
from torch import nn

import test_transformer
# 输入参数：
#    tokens_a, tokens_b - ⼀个句⼦或两个句⼦
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
	"""获取输⼊序列的词元及其⽚段索引"""
	tokens = ['<cls>'] + tokens_a + ['<sep>']
	# 0和1分别标记⽚段A和B
	segments = [0] * (len(tokens_a) + 2)
	if tokens_b is not None:
		# BERT输⼊序列的标记
		tokens += tokens_b + ['<sep>']
		# 相应的⽚段索引。
		segments += [1] * (len(tokens_b) + 1)
	# 返回BERT输⼊序列的标记及其相应的⽚段索引。
	return tokens, segments

# 与TransformerEncoder不同，BERTEncoder使⽤⽚段嵌⼊和可学习的位置嵌⼊。
#@save
class BERTEncoder(nn.Module):
	"""BERT编码器"""
	def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
				ffn_num_hiddens, num_heads, num_layers, dropout,
				max_len=1000, key_size=768, query_size=768, value_size=768, 
				**kwargs):
		super(BERTEncoder, self).__init__(**kwargs)
		self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
		self.segment_embedding = nn.Embedding(2, num_hiddens)
		self.blks = nn.Sequential()
		for i in range(num_layers):
			self.blks.add_module(f"{i}", test_transformer.EncoderBlock(
				key_size, query_size, value_size, num_hiddens, norm_shape,
				ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
		# 在BERT中，位置嵌⼊是可学习的，因此我们创建⼀个⾜够⻓的位置嵌⼊参数
		self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
				num_hiddens))
	def forward(self, tokens, segments, valid_lens):
		# 在以下代码段中，X的形状保持不变：（批量⼤⼩，最⼤序列⻓度，num_hiddens） 
		X = self.token_embedding(tokens) + self.segment_embedding(segments)
		X = X + self.pos_embedding.data[:, :X.shape[1], :]
		for blk in self.blks:
			X = blk(X, valid_lens)
		return X

if __name__ == '__main__':
	# 假设词表⼤⼩为10000，为了演⽰BERTEncoder的前向推断，让我们创建⼀个实例并初始化它的参数。
	vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
	norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
	encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
			ffn_num_hiddens, num_heads, num_layers, dropout)

	# 我们将tokens定义为⻓度为8的2个输⼊序列，其中每个词元是词表的索引。
	# 使⽤输⼊tokens的BERTEncoder的前向推断返回编码结果，
	# 其中每个词元由向量表⽰，其⻓度由超参数num_hiddens定义。
	# 此超参数通常称为Transformer编码器的隐藏⼤⼩（隐藏单元数）。
	tokens = torch.randint(0, vocab_size, (2, 8))
	segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
	encoded_X = encoder(tokens, segments, None)
	print("encoded_X.shape : ", encoded_X.shape)
# 掩蔽语⾔模型：
# 我们实现了下⾯的MaskLM类来预测BERT预训练的掩蔽语⾔模型任务中的掩蔽标记。
# 预测使⽤单隐藏层的多层感知机（self.mlp）。
#@save
class MaskLM(nn.Module):
	"""BERT的掩蔽语⾔模型任务"""
	def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
		super(MaskLM, self).__init__(**kwargs)
		self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
								nn.ReLU(),
								nn.LayerNorm(num_hiddens),
								nn.Linear(num_hiddens, vocab_size))
	# 在前向推断中，它需要两个输⼊：
	# BERTEncoder的编码结果和⽤于预测的词元位置。
	def forward(self, X, pred_positions):
		num_pred_positions = pred_positions.shape[1]
		pred_positions = pred_positions.reshape(-1)
		batch_size = X.shape[0]
		batch_idx = torch.arange(0, batch_size)
		# 假设batch_size=2，num_pred_positions=3
		# 那么batch_idx是np.array（[0,0,0,1,1]）
		batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
		masked_X = X[batch_idx, pred_positions]
		masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
		mlm_Y_hat = self.mlp(masked_X)
		# 输出是这些位置的预测结果。
		return mlm_Y_hat

if __name__ == '__main__':
	mlm = MaskLM(vocab_size, num_hiddens)
	# 来⾃BERTEncoder的正向推断encoded_X表⽰2个BERT输⼊序列。
	# 我们将mlm_positions定义为在encoded_X的任⼀输⼊序列中预测的3个指⽰。
	mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
	# mlm的前向推断返回encoded_X的所有掩蔽位置mlm_positions处的预测结果mlm_Y_hat。
	mlm_Y_hat = mlm(encoded_X, mlm_positions)
	# 对于每个预测，结果的⼤⼩等于词表的⼤⼩。
	print("mlm_Y_hat.shape : ", mlm_Y_hat.shape)

	mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
	loss = nn.CrossEntropyLoss(reduction='none')
	# 通过掩码下的预测词元mlm_Y的真实标签mlm_Y_hat，
	# 我们可以计算在BERT预训练中的遮蔽语⾔模型任务的交叉熵损失。
	mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
	print("mlm_l.shape : ", mlm_l.shape)

# 下⼀句预测：
#    为了帮助理解两个⽂本序列之间的关系，BERT在预训练中考虑了⼀个⼆元分类任务——下⼀句预测。
#    在为预训练⽣成句⼦对时，有⼀半的时间它们确实是标签为“真”的连续句⼦；
#    在另⼀半的时间⾥，第⼆个句⼦是从语料库中随机抽取的，标记为“假”。
#@save
class NextSentencePred(nn.Module):
	"""BERT的下⼀句预测任务"""
	def __init__(self, num_inputs, **kwargs):
		super(NextSentencePred, self).__init__(**kwargs)
		# 使⽤单隐藏层的多层感知机来预测
		# 第⼆个句⼦是否是BERT输⼊序列中第⼀个句⼦的下⼀个句⼦。
		self.output = nn.Linear(num_inputs, 2)
	# 由于Transformer编码器中的⾃注意⼒，特殊词元“<cls>”的BERT表⽰已经对输⼊的两个句⼦进⾏了编码。
	# 因此，多层感知机分类器的输出层（self.output）以X作为输⼊，
	# 其中X是多层感知机隐藏层的输出，⽽MLP隐藏层的输⼊是编码后的“<cls>”词元。
	def forward(self, X):
		# X的形状：(batchsize,num_hiddens)
		return self.output(X)

if __name__ == '__main__':
	# 我们可以看到，NextSentencePred实例的前向推断返回每个BERT输⼊序列的⼆分类预测。
	encoded_X = torch.flatten(encoded_X, start_dim=1)
	# NSP的输⼊形状:(batchsize，num_hiddens)
	nsp = NextSentencePred(encoded_X.shape[-1])
	nsp_Y_hat = nsp(encoded_X)
	print("nsp_Y_hat.shape : ", nsp_Y_hat.shape)

	# 还可以计算两个⼆元分类的交叉熵损失。
	nsp_y = torch.tensor([0, 1])
	nsp_l = loss(nsp_Y_hat, nsp_y)
	print("nsp_l.shape : ", nsp_l.shape)

# 在预训练BERT时，最终的损失函数是掩蔽语⾔模型损失函数和下⼀句预测损失函数的线性组合。
#@save
class BERTModel(nn.Module):
	"""BERT模型"""
	def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
				ffn_num_hiddens, num_heads, num_layers, dropout,
				max_len=1000, key_size=768, query_size=768, value_size=768,
				hid_in_features=768, mlm_in_features=768,
				nsp_in_features=768):
		super(BERTModel, self).__init__()
		# 现在我们可以通过实例化三个类BERTEncoder、
		# MaskLM和NextSentencePred来定义BERTModel类。
		self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
				ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
				dropout, max_len=max_len, key_size=key_size,
				query_size=query_size, value_size=value_size)
		self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
				nn.Tanh())
		self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
		self.nsp = NextSentencePred(nsp_in_features)
	def forward(self, tokens, segments, valid_lens=None,
				pred_positions=None):
		encoded_X = self.encoder(tokens, segments, valid_lens)
		if pred_positions is not None:
			mlm_Y_hat = self.mlm(encoded_X, pred_positions)
		else:
			mlm_Y_hat = None
		# ⽤于下⼀句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
		nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
		# 前向推断返回编码后的BERT表⽰encoded_X、
		# 掩蔽语⾔模型预测mlm_Y_hat和下⼀句预测nsp_Y_hat。
		return encoded_X, mlm_Y_hat, nsp_Y_hat



