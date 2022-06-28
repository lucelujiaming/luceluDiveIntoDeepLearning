import torch
from torch import nn

import test_attention_scoring_functions
import test_seq2seq
import my_encoder_decoder
import train_framework
import test_attention_cues
import test_machine_translation_and_dataset

import my_encoder_decoder
import test_attention_cues

# 其实，我们只需重新定义解码器即可。为了更⽅便地显⽰学习的注意⼒权重，
# 以下AttentionDecoder类定义了带有注意⼒机制解码器的基本接⼝。
#@save
class AttentionDecoder(my_encoder_decoder.Decoder):
	"""带有注意⼒机制解码器的基本接⼝"""
	def __init__(self, **kwargs):
		super(AttentionDecoder, self).__init__(**kwargs)
	@property
	def attention_weights(self):
		raise NotImplementedError

# 让我们在接下来的Seq2SeqAttentionDecoder类中实现带有Bahdanau注意⼒的循环神经⽹络解码器。
# 在每个解码时间步骤中，解码器上⼀个时间步的最终层隐状态将⽤作查询。
class Seq2SeqAttentionDecoder(AttentionDecoder):
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
						dropout=0, **kwargs):
		super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
		# 加性注意⼒。
		self.attention = test_attention_scoring_functions.AdditiveAttention(
					num_hiddens, num_hiddens, num_hiddens, dropout)
		# 嵌⼊层
		# 我们使⽤了嵌⼊层（embedding layer）来获得输⼊序列中每个词元的特征向量。
		# 嵌⼊层的权重是⼀个矩阵，其⾏数等于输⼊词表的⼤⼩（vocab_size），
		# 其列数等于特征向量的维度（embed_size）。
		self.embedding = nn.Embedding(vocab_size, embed_size)
		# 本⽂选择了⼀个多层⻔控循环单元来实现编码器。参见test_seq2seq。
		# 因此，注意⼒输出和输⼊嵌⼊都连结为循环神经⽹络解码器的输⼊。
		self.rnn = nn.GRU(
				embed_size + num_hiddens, num_hiddens, num_layers,
				dropout=dropout)
		self.dense = nn.Linear(num_hiddens, vocab_size)

	def init_state(self, enc_outputs, enc_valid_lens, *args):
		# ⾸先，我们初始化解码器的状态，需要下⾯的输⼊：
		#   1. 编码器在所有时间步的最终层隐状态，将作为注意⼒的键和值；
		#   2. 上⼀时间步的编码器全层隐状态，将作为初始化解码器的隐状态；
		#   3. 编码器有效⻓度（排除在注意⼒池中填充词元）。
		# outputs的形状为(batch_size，num_steps，num_hiddens).
		# hidden_state的形状为(num_layers，batch_size，num_hiddens)
		outputs, hidden_state = enc_outputs
		return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

	def forward(self, X, state):
		# enc_outputs的形状为(batch_size,num_steps,num_hiddens).
		# hidden_state的形状为(num_layers,batch_size,
		# num_hiddens)
		enc_outputs, hidden_state, enc_valid_lens = state
		# 输出X的形状为(num_steps,batch_size,embed_size)
		X = self.embedding(X).permute(1, 0, 2)
		outputs, self._attention_weights = [], []
		for x in X:
			# query的形状为(batch_size,1,num_hiddens)
			query = torch.unsqueeze(hidden_state[-1], dim=1)
			# context的形状为(batch_size,1,num_hiddens)
			context = self.attention(
				query, enc_outputs, enc_outputs, enc_valid_lens)
			# 在特征维度上连结
			x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1) 
			# 将x变形为(1,batch_size,embed_size+num_hiddens)
			out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
			outputs.append(out)
			self._attention_weights.append(self.attention.attention_weights)
		# 全连接层变换后，outputs的形状为
		# (num_steps,batch_size,vocab_size)
		outputs = self.dense(torch.cat(outputs, dim=0))
		return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
		enc_valid_lens]
	@property
	def attention_weights(self):
		return self._attention_weights

if __name__ == '__main__':
	encoder = test_seq2seq.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
								num_layers=2)
	encoder.eval()
	decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
								num_layers=2)
	decoder.eval()
	# 我们使⽤包含7个时间步的4个序列输⼊的⼩批量测试Bahdanau注意⼒解码器。
	X = torch.zeros((4, 7), dtype=torch.long) # (batch_size,num_steps)
	state = decoder.init_state(encoder(X), None)
	output, state = decoder(X, state)
	print("output.shape : ", output.shape)
	print("len(state) : ", len(state))
	print("state[0].shape : ", state[0].shape)
	print("len(state[1]) : ", len(state[1]))
	print("state[1][0].shape : ", state[1][0].shape)

	# 我们在这⾥指定超参数，实例化⼀个带有Bahdanau注意⼒的编码器和解码器，并对这个模型进⾏机器翻译训练。
	embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
	batch_size, num_steps = 64, 10
	lr, num_epochs, device = 0.005, 250, train_framework.try_gpu()

	train_iter, src_vocab, tgt_vocab = \
			test_machine_translation_and_dataset.load_data_nmt(batch_size, num_steps)
	encoder = test_seq2seq.Seq2SeqEncoder(
				len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
	decoder = Seq2SeqAttentionDecoder(
				len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
	net = my_encoder_decoder.EncoderDecoder(encoder, decoder)
	test_seq2seq.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

	# 模型训练后，我们⽤它将⼏个英语句⼦翻译成法语并计算它们的BLEU分数。
	engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
	fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
	for eng, fra in zip(engs, fras):
		translation, dec_attention_weight_seq = test_seq2seq.predict_seq2seq(
				net, eng, src_vocab, tgt_vocab, num_steps, device, True)
		print(f'{eng} => {translation}, ',
				f'bleu {test_seq2seq.bleu(translation, fra, k=2):.3f}')

	attention_weights = torch.cat([step[0][0][0] 
		for step in dec_attention_weight_seq], 0).reshape((1, 1, -1, num_steps))

	# 训练结束后，下⾯我们通过可视化注意⼒权重你会发现，每个查询都会在键值对上分配不同的权重，
	# 这说明在每个解码步中，输⼊序列的不同部分被选择性地聚集在注意⼒池中。
	# 加上⼀个包含序列结束词元
	test_attention_cues.show_heatmaps(
		attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
		xlabel='Key positions', ylabel='Query positions', usingION = False)
	my_plt.plt.ioff()
	my_plt.plt.show()







