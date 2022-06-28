import math
import pandas as pd
import torch
from torch import nn

import train_framework
import my_encoder_decoder
import test_multihead_attention
import test_attention_cues
import my_plt
import test_positional_encoding
import test_seq2seq
import test_machine_translation_and_dataset
import test_bahdanau_attention

# 基于位置的前馈⽹络对序列中的所有位置的表⽰进⾏变换时使⽤的是同⼀个多层感知机。
#@save
class PositionWiseFFN(nn.Module):
	"""基于位置的前馈⽹络"""
	def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
						**kwargs):
		super(PositionWiseFFN, self).__init__(**kwargs)
		# 一个全连接层，接一个ReLU层，接一个全连接层。
		self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
		self.relu = nn.ReLU()
		self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
	def forward(self, X):
		# 一个全连接层，接一个ReLU层，接一个全连接层。
		return self.dense2(self.relu(self.dense1(X)))

if __name__ == '__main__':
	# 下⾯的例⼦显⽰，改变张量的最⾥层维度的尺⼨，会改变成基于位置的前馈⽹络的输出尺⼨。
	ffn = PositionWiseFFN(4, 4, 8)
	ffn.eval()
	print("ffn(torch.ones((2, 3, 4)))[0] : ", 
		ffn(torch.ones((2, 3, 4)))[0])

	# 以下代码对⽐不同维度的层规范化和批量规范化的效果。
	ln = nn.LayerNorm(2)
	bn = nn.BatchNorm1d(2) 
	X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
	# 在训练模式下计算X的均值和⽅差
	print('layer norm:', ln(X), '\nbatch norm:', bn(X))

# 现在我们可以使⽤残差连接和层规范化来实现AddNorm类。暂退法也被作为正则化⽅法使⽤。
#@save
class AddNorm(nn.Module):
	"""残差连接后进⾏层规范化"""
	def __init__(self, normalized_shape, dropout, **kwargs):
		super(AddNorm, self).__init__(**kwargs)
		self.dropout = nn.Dropout(dropout)
		self.ln = nn.LayerNorm(normalized_shape)
	def forward(self, X, Y):
		# 使⽤残差连接。残差⽹络的核⼼思想是：
		# 每个附加层都应该更容易地包含原始函数作为其元素之⼀。
		return self.ln(self.dropout(Y) + X)

if __name__ == '__main__':
	add_norm = AddNorm([3, 4], 0.5)
	add_norm.eval()
	shapeRet = add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape
	print("add_norm().shapeRet : ", shapeRet)

# 从宏观⻆度来看，transformer的编码器是由多个相同的层叠加⽽成的，
# 每个层都有两个⼦层（⼦层表⽰为sublayer）。
#     第⼀个⼦层是多头⾃注意⼒（multi-head self-attention）汇聚；
#     第⼆个⼦层是基于位置的前馈⽹络（positionwise feed-forward network）

# 有了组成transformer编码器的基础组件，现在可以先实现编码器中的⼀个层。
# 下⾯的EncoderBlock类包含两个⼦层：多头⾃注意⼒和基于位置的前馈⽹络，
# 这两个⼦层都使⽤了残差连接和紧随的层规范化。
#@save
class EncoderBlock(nn.Module):
	"""transformer编码器块"""
	def __init__(self, key_size, query_size, value_size, num_hiddens,
			norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
			dropout, use_bias=False, **kwargs):
		super(EncoderBlock, self).__init__(**kwargs)
		# 第⼀个⼦层是多头⾃注意⼒（multi-head self-attention）汇聚；
		self.attention = test_multihead_attention.MultiHeadAttention(
				key_size, query_size, value_size, 
				num_hiddens, num_heads, dropout,
				use_bias)
		# AddNorm在残差连接后进⾏层规范化：
		# 在残差连接的加法计算之后，紧接着应⽤层规范化（layer normalization）。
		self.addnorm1 = AddNorm(norm_shape, dropout)
		# 第⼆个⼦层是基于位置的前馈⽹络（positionwise feed-forward network）
		self.ffn = PositionWiseFFN(
				ffn_num_input, ffn_num_hiddens, num_hiddens)
		# 在残差连接的加法计算之后，紧接着应⽤层规范化（layer normalization）。
		self.addnorm2 = AddNorm(norm_shape, dropout)
	def forward(self, X, valid_lens):
		Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
		return self.addnorm2(Y, self.ffn(Y))

if __name__ == '__main__':
	# 正如我们所看到的，transformer编码器中的任何层都不会改变其输⼊的形状。
	X = torch.ones((2, 100, 24))
	valid_lens = torch.tensor([3, 2])
	encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
	encoder_blk.eval()
	print("encoder_blk(X, valid_lens).shape : ", encoder_blk(X, valid_lens).shape)

#@save
class TransformerEncoder(my_encoder_decoder.Encoder):
	"""transformer编码器"""
	def __init__(self, vocab_size, key_size, query_size, value_size,
			num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
			num_heads, num_layers, dropout, use_bias=False, **kwargs):
		super(TransformerEncoder, self).__init__(**kwargs)
		self.num_hiddens = num_hiddens
		self.embedding = nn.Embedding(vocab_size, num_hiddens)
		self.pos_encoding = \
			test_positional_encoding.PositionalEncoding(num_hiddens, dropout)
		self.blks = nn.Sequential()
		# 在实现下⾯的transformer编码器的代码中，我们堆叠了num_layers个EncoderBlock类的实例。
		for i in range(num_layers):
			self.blks.add_module("block"+str(i),
				EncoderBlock(key_size, query_size, value_size, num_hiddens,
					norm_shape, ffn_num_input, ffn_num_hiddens,
					num_heads, dropout, use_bias))

	def forward(self, X, valid_lens, *args):
		# 由于我们使⽤的是值范围在−1和1之间的固定位置编码，
		# 因此通过学习得到的输⼊的嵌⼊表⽰的值需要先乘以嵌⼊维度的平⽅根进⾏重新缩放，
		# 然后再与位置编码相加。
		# 因为位置编码值在-1和1之间，因此嵌⼊值乘以嵌⼊维度的平⽅根进⾏缩放，
		# 然后再与位置编码相加。
		X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
		self.attention_weights = [None] * len(self.blks)
		for i, blk in enumerate(self.blks):
			X = blk(X, valid_lens)
			self.attention_weights[
				i] = blk.attention.attention.attention_weights
		return X

if __name__ == '__main__':
	# 下⾯我们指定了超参数来创建⼀个两层的transformer编码器。
	# Transformer编码器输出的形状是（批量⼤⼩，时间步数⽬，num_hiddens）。
	encoder = TransformerEncoder(
			200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
	encoder.eval()
	shapeRet = encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape
	print("encoder().shapeRet : ", shapeRet)

# transformer解码器也是由多个相同的层组成。在DecoderBlock类中实现的每个层包含了三个⼦层：
# 解码器⾃注意⼒、“编码器-解码器”注意⼒和基于位置的前馈⽹络。
# 这些⼦层也都被残差连接和紧随的层规范化围绕。
class DecoderBlock(nn.Module):
	"""解码器中第i个块"""
	def __init__(self, key_size, query_size, value_size, num_hiddens,
			norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
			dropout, i, **kwargs):
		super(DecoderBlock, self).__init__(**kwargs)
		self.i = i
		#  第⼀个⼦层是多头⾃注意⼒（multi-head self-attention）汇聚；
		self.attention1 = test_multihead_attention.MultiHeadAttention(
			key_size, query_size, value_size, num_hiddens, num_heads, dropout)
		# 在残差连接的加法计算之后，紧接着应⽤层规范化（layer normalization）。
		self.addnorm1 = AddNorm(norm_shape, dropout)
		# 第⼆个⼦层是基于位置的前馈⽹络。称为编码器－解码器注意⼒层。
		self.attention2 = test_multihead_attention.MultiHeadAttention(
			key_size, query_size, value_size, num_hiddens, num_heads, dropout)
		# 在残差连接的加法计算之后，紧接着应⽤层规范化（layer normalization）。
		self.addnorm2 = AddNorm(norm_shape, dropout)
		# 第三个⼦层是基于位置的前馈⽹络（positionwise feed-forward network）
		self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
						num_hiddens)
		# 在残差连接的加法计算之后，紧接着应⽤层规范化（layer normalization）。
		self.addnorm3 = AddNorm(norm_shape, dropout)

	def forward(self, X, state):
		enc_outputs, enc_valid_lens = state[0], state[1] 
		# 训练阶段，输出序列的所有词元都在同⼀时间处理，
		# 因此state[2][self.i]初始化为None。 
		# 预测阶段，输出序列是通过词元⼀个接着⼀个解码的，
		# 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表⽰
		if state[2][self.i] is None:
			# 如果是训练阶段
			key_values = X
		else:
			# 如果是预测阶段
			key_values = torch.cat((state[2][self.i], X), axis=1)
		state[2][self.i] = key_values
		if self.training:
			# 如果是训练阶段：计算dec_valid_lens。
			batch_size, num_steps, _ = X.shape
			# dec_valid_lens的开头:(batch_size,num_steps),
			# 其中每⼀⾏是[1,2,...,num_steps]
			dec_valid_lens = torch.arange(
				1, num_steps + 1, device=X.device).repeat(batch_size, 1)
		else:
			# 如果是预测阶段：不需要计算dec_valid_lens。
			dec_valid_lens = None
		# ⾃注意⼒
		X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
		Y = self.addnorm1(X, X2)
		# 编码器－解码器注意⼒。
		# enc_outputs的开头:(batch_size,num_steps,num_hiddens)

		# 为了在解码器中保留⾃回归的属性，其掩蔽⾃注意⼒设定了参数dec_valid_lens，
		# 以便任何查询都只会与解码器中所有已经⽣成词元的位置
		# （即直到该查询位置为⽌）进⾏注意⼒计算。
		Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
		Z = self.addnorm2(Y, Y2)
		return self.addnorm3(Z, self.ffn(Z)), state

if __name__ == '__main__':
	# 为了便于在“编码器－解码器”注意⼒中进⾏缩放点积计算和残差连接中进⾏加法计算，
	# 编码器和解码器的特征维度都是num_hiddens。
	decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
	decoder_blk.eval()
	X = torch.ones((2, 100, 24))
	state = [encoder_blk(X, valid_lens), valid_lens, [None]]
	decoder_blk(X, state)[0].shape

# 解码器的⾃注意⼒权重和编码器解码器注意⼒权重都被存储下来，⽅便⽇后可视化的需要。
class TransformerDecoder(test_bahdanau_attention.AttentionDecoder):
	def __init__(self, vocab_size, key_size, query_size, value_size,
		num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
		num_heads, num_layers, dropout, **kwargs):
		super(TransformerDecoder, self).__init__(**kwargs)
		self.num_hiddens = num_hiddens
		self.num_layers = num_layers
		self.embedding = nn.Embedding(vocab_size, num_hiddens)
		self.pos_encoding = test_positional_encoding.PositionalEncoding(num_hiddens, dropout)
		self.blks = nn.Sequential()
		# 现在我们构建了由num_layers个DecoderBlock实例组成的完整的transformer解码器。
		for i in range(num_layers):
			self.blks.add_module("block"+str(i),
				DecoderBlock(key_size, query_size, value_size, num_hiddens,
					norm_shape, ffn_num_input, ffn_num_hiddens,
					num_heads, dropout, i))
		# 最后，通过⼀个全连接层计算所有vocab_size个可能的输出词元的预测值。
		self.dense = nn.Linear(num_hiddens, vocab_size)
	def init_state(self, enc_outputs, enc_valid_lens, *args):
		return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
	def forward(self, X, state):
		X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
		self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
		for i, blk in enumerate(self.blks):
			X, state = blk(X, state)
			# 解码器⾃注意⼒权重被存储下来，⽅便⽇后可视化的需要。
			self._attention_weights[0][
				i] = blk.attention1.attention.attention_weights
			# “编码器－解码器”⾃注意⼒权重被存储下来，⽅便⽇后可视化的需要。
			self._attention_weights[1][
				i] = blk.attention2.attention.attention_weights
		return self.dense(X), state
	@property
	def attention_weights(self):
		return self._attention_weights

if __name__ == '__main__':
	# 依照transformer架构来实例化编码器－解码器模型。
	# 在这⾥，指定transformer的编码器和解码器都是2层，都使⽤4头注意⼒。
	num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
	lr, num_epochs, device = 0.005, 200, train_framework.try_gpu()
	ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
	key_size, query_size, value_size = 32, 32, 32
	norm_shape = [32]

if __name__ == '__main__':
	# 为了进⾏序列到序列的学习，我们在“英语－法语”机器翻译数据集上训练transformer模型。
	train_iter, src_vocab, tgt_vocab = \
		test_machine_translation_and_dataset.load_data_nmt(batch_size, num_steps)
	encoder = TransformerEncoder(
		len(src_vocab), key_size, query_size, value_size, num_hiddens,
		norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
		num_layers, dropout)

	decoder = TransformerDecoder(
			len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
			norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
			num_layers, dropout)
	net = my_encoder_decoder.EncoderDecoder(encoder, decoder)
	test_seq2seq.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

if __name__ == '__main__':
	# 训练结束后，使⽤transformer模型将⼀些英语句⼦翻译成法语，并且计算它们的BLEU分数。
	engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
	fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
	for eng, fra in zip(engs, fras):
		translation, dec_attention_weight_seq = test_seq2seq.predict_seq2seq(
			net, eng, src_vocab, tgt_vocab, num_steps, device, True)
		print(f'{eng} => {translation}, ', f'bleu {test_seq2seq.bleu(translation, fra, k=2):.3f}')

	# 当进⾏最后⼀个英语到法语的句⼦翻译⼯作时，让我们可视化transformer的注意⼒权重。
	enc_attention_weights = torch.cat(
		net.encoder.attention_weights, 0).reshape((num_layers, num_heads,
			-1, num_steps))
	print("enc_attention_weights.shape : ", enc_attention_weights.shape)

	# 接下来，将逐⾏呈现两层多头注意⼒的权重。每个注意⼒头都根据查询、键和值的不同的表⽰⼦空间来表⽰不同的注意⼒。
	test_attention_cues.show_heatmaps(
		enc_attention_weights.cpu(), xlabel='Key positions',
		ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
		figsize=(7, 3.5), usingION = False)
	my_plt.plt.ioff()
	my_plt.plt.show()

if __name__ == '__main__':
	# 为了可视化解码器的⾃注意⼒权重和“编码器－解码器”的注意⼒权重，我们需要完成更多的数据操作⼯作。
	dec_attention_weights_2d = [head[0].tolist()
				for step in dec_attention_weight_seq
				for attn in step for blk in attn for head in blk]
	dec_attention_weights_filled = torch.tensor(
				# 例如，我们⽤零填充被掩蔽住的注意⼒权重。
				pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
	dec_attention_weights = dec_attention_weights_filled.reshape(
				(-1, 2, num_layers, num_heads, num_steps))

	dec_self_attention_weights, dec_inter_attention_weights = \
	dec_attention_weights.permute(1, 2, 3, 0, 4)
	dec_self_attention_weights.shape, dec_inter_attention_weights.shape

if __name__ == '__main__':
	# 由于解码器⾃注意⼒的⾃回归属性，查询不会对当前位置之后的“键－值”对进⾏注意⼒计算。
	# Plusonetoincludethebeginning-of-sequencetoken
	test_attention_cues.show_heatmaps(
		dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
		xlabel='Key positions', ylabel='Query positions',
		titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5), usingION = False)
	my_plt.plt.ioff()
	my_plt.plt.show()

if __name__ == '__main__':
	# 与编码器的⾃注意⼒的情况类似，通过指定输⼊序列的有效⻓度，
	# 输出序列的查询不会与输⼊序列中填充位置的词元进⾏注意⼒计算。
	test_attention_cues.show_heatmaps(
		dec_inter_attention_weights, xlabel='Key positions',
		ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
		figsize=(7, 3.5), usingION = False)
	my_plt.plt.ioff()
	my_plt.plt.show()



