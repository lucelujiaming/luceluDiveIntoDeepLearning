import collections
import math
import torch
from torch import nn

import train_framework
import my_data_time_machine
import my_encoder_decoder
import my_timer
import test_machine_translation_and_dataset

#@save
class Seq2SeqEncoder(my_encoder_decoder.Encoder):
	"""⽤于序列到序列学习的循环神经⽹络编码器"""
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
					dropout=0, **kwargs):
		super(Seq2SeqEncoder, self).__init__(**kwargs)
		# 嵌⼊层
		# 我们使⽤了嵌⼊层（embedding layer）来获得输⼊序列中每个词元的特征向量。
		# 嵌⼊层的权重是⼀个矩阵，其⾏数等于输⼊词表的⼤⼩（vocab_size），
		# 其列数等于特征向量的维度（embed_size）。
		self.embedding = nn.Embedding(vocab_size, embed_size)
		# 本⽂选择了⼀个多层⻔控循环单元来实现编码器。
		self.rnn = nn.GRU(
			# 输入数据X的特征值的数目。
			embed_size, 
			# 隐藏层的神经元数量，也就是隐藏层的特征数量。
			num_hiddens, 
			# 循环神经网络的层数，默认值是 1。
			num_layers,
			# bias：默认为True，如果为false则表示神经元不使用bias(ih)和bias(hh)偏移参数。
			# batch_first：如果设置为True，则输入数据的维度中第一个维度就是batch值，
			#              默认为False。默认情况下第一个维度是序列的长度，
			#              第二个维度才是 - batch，第三个维度是特征数目。

			# 如果不为空，则表示最后跟一个 dropout 层抛弃部分数据，
			# 抛弃数据的比例由该参数指定。默认为0。
			dropout=dropout)
	def forward(self, X, *args):
		# 输出'X'的形状：(batch_size,num_steps,embed_size)
		X = self.embedding(X)
		# 在循环神经⽹络模型中，第⼀个轴对应于时间步
		X = X.permute(1, 0, 2) # 如果未提及状态，则默认为0
		output, state = self.rnn(X)
		# output的形状:(num_steps,batch_size,num_hiddens)
		# state[0]的形状:(num_layers,batch_size,num_hiddens)
		return output, state

if __name__ == '__main__':
	# 使⽤⼀个两层⻔控循环单元编码器，其隐藏单元数为16。
	encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
					num_layers=2)
	encoder.eval()
	# 给定⼀⼩批量的输⼊序列X（批量⼤⼩为4，时间步为7）。
	X = torch.zeros((4, 7), dtype=torch.long)
	output, state = encoder(X)
	# 在完成所有时间步后，最后⼀层的隐状态的输出是⼀个张量
	# （output由编码器的循环层返回），其形状为（时间步数，批量⼤⼩，隐藏单元数）。
	print("output.shape : ", output.shape)
	# 在最后⼀个时间步的多层隐状态的形状是
	# （隐藏层的数量，批量⼤⼩，隐藏单元的数量）。
	# 如果使⽤⻓短期记忆⽹络，state中还将包含记忆单元信息。
	print("state.shape : ", state.shape)

class Seq2SeqDecoder(my_encoder_decoder.Decoder):
	"""⽤于序列到序列学习的循环神经⽹络解码器"""
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
				dropout=0, **kwargs):
		super(Seq2SeqDecoder, self).__init__(**kwargs)
		# 其为一个简单的存储固定大小的词典的嵌入向量的查找表。
		self.embedding = nn.Embedding(
			# 词典的大小尺寸
			vocab_size, 
			# 嵌入向量的维度
			embed_size)
		# 为了进⼀步包含经过编码的输⼊序列的信息，
		# 上下⽂变量在所有的时间步与解码器的输⼊进⾏拼接（concatenate）。
		self.rnn = nn.GRU(
			# 输入序列的一维向量的长度。
			embed_size + num_hiddens, 
			# 隐层的输出特征的长度。
			num_hiddens, 
			# 隐藏层堆叠的高度，用于增加隐层的深度。
			num_layers,
			# 默认0 若非0，则为dropout率。
			dropout=dropout)
		# 为了预测输出词元的概率分布，
		# 在循环神经⽹络解码器的最后⼀层使⽤全连接层来变换隐状态。
		self.dense = nn.Linear(num_hiddens, vocab_size)
	def init_state(self, enc_outputs, *args):
		# 当实现解码器时，我们直接使⽤编码器最后⼀个时间步的隐状态来初始化解码器的隐状态。
		return enc_outputs[1]
	def forward(self, X, state):
		# 输出'X'的形状：(batch_size,num_steps,embed_size)
		X = self.embedding(X).permute(1, 0, 2) 
		# 这就要求使⽤循环神经⽹络实现的编码器和解码器具有相同数量的层和隐藏单元。
		# ⼴播context，使其具有与X相同的num_steps
		context = state[-1].repeat(X.shape[0], 1, 1)
		X_and_context = torch.cat((X, context), 2)
		output, state = self.rnn(X_and_context, state)
		output = self.dense(output).permute(1, 0, 2)
		# output的形状:(batch_size,num_steps,vocab_size)
		# state[0]的形状:(num_layers,batch_size,num_hiddens)
		return output, state

if __name__ == '__main__':
	# 我们⽤与前⾯提到的编码器中相同的超参数来实例化解码器。
	# 如我们所⻅，解码器的输出形状变为（批量⼤⼩，时间步数，词表⼤⼩），
	# 其中张量的最后⼀个维度存储预测的词元分布。
	decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
								num_layers=2)
	decoder.eval()
	state = decoder.init_state(encoder(X))
	output, state = decoder(X, state)
	print("output.shape, state.shape : ", output.shape, state.shape)

# 通过零值化屏蔽不相关的项，以便后⾯任何不相关预测的计算都是与零的乘积，结果都等于零。
#@save
def sequence_mask(X, valid_len, value=0):
	"""在序列中屏蔽不相关的项"""
	maxlen = X.size(1)
	mask = torch.arange((maxlen), dtype=torch.float32,
					device=X.device)[None, :] < valid_len[:, None]
	X[~mask] = value
	return X

if __name__ == '__main__':
	X = torch.tensor([[1, 2, 3], [4, 5, 6]])
	print("sequence_mask(X, torch.tensor([1, 2])) : ", 
		sequence_mask(X, torch.tensor([1, 2])))
	# 我们还可以使⽤此函数屏蔽最后⼏个轴上的所有项。
	X = torch.ones(2, 3, 4)
	print("sequence_mask(X, torch.tensor([1, 2]), value=-1) : ", 
		sequence_mask(X, torch.tensor([1, 2]), value=-1))

# 我们可以通过扩展softmax交叉熵损失函数来遮蔽不相关的预测。
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
	"""带遮蔽的softmax交叉熵损失函数"""
	# pred的形状：(batch_size,num_steps,vocab_size)
	# label的形状：(batch_size,num_steps)
	# valid_len的形状：(batch_size,)
	# 我们可以通过扩展softmax交叉熵损失函数来遮蔽不相关的预测。
	def forward(self, pred, label, valid_len):
		# 最初，所有预测词元的掩码都设置为1。
		weights = torch.ones_like(label)
		# ⼀旦给定了有效⻓度，与填充词元对应的掩码将被设置为0。
		weights = sequence_mask(weights, valid_len)
		self.reduction='none'
		unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
					pred.permute(0, 2, 1), label)
		# 最后，将所有词元的损失乘以掩码，以过滤掉损失中填充词元产⽣的不相关预测。
		weighted_loss = (unweighted_loss * weights).mean(dim=1)
		return weighted_loss

if __name__ == '__main__':
	loss = MaskedSoftmaxCELoss()
	# 以创建三个相同的序列来进⾏代码健全性检查，然后分别指定这些序列的有效⻓度为4、2和0。
	lossRet = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
					torch.tensor([4, 2, 0]))
	# 结果就是，第⼀个序列的损失应为第⼆个序列的两倍，⽽第三个序列的损失应为零。
	# lossRet :  tensor([2.3026, 1.1513, 0.0000])
	print("lossRet : ", lossRet)

#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
	"""训练序列到序列模型"""
	def xavier_init_weights(m):
		if type(m) == nn.Linear:
			nn.init.xavier_uniform_(m.weight)
		if type(m) == nn.GRU:
			for param in m._flat_weights_names:
				if "weight" in param:
					nn.init.xavier_uniform_(m._parameters[param])
	net.apply(xavier_init_weights)
	net.to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	loss = MaskedSoftmaxCELoss()
	net.train()
	animator = train_framework.Animator(xlabel='epoch', ylabel='loss',
					xlim=[10, num_epochs])
	for epoch in range(num_epochs):
		timer = my_timer.Timer()
		metric = train_framework.Accumulator(2) # 训练损失总和，词元数量
		for batch in data_iter:
			optimizer.zero_grad()
			X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
			# 特定的序列开始词元（“<bos>”）和原始的输出序列
			# （不包括序列结束词元“<eos>”）拼接在⼀起作为解码器的输⼊。
			bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
					device=device).reshape(-1, 1)
			# 这被称为强制教学（teacher forcing），
			# 因为原始的输出序列（词元的标签）被送⼊解码器。
			# 或者，将来⾃上⼀个时间步的预测得到的词元作为解码器的当前输⼊。
			dec_input = torch.cat([bos, Y[:, :-1]], 1) # 强制教学
			Y_hat, _ = net(X, dec_input, X_valid_len)
			l = loss(Y_hat, Y, Y_valid_len)
			l.sum().backward() # 损失函数的标量进⾏“反向传播”
			train_framework.grad_clipping(net, 1)

			num_tokens = Y_valid_len.sum()
			optimizer.step()
			with torch.no_grad():
				metric.add(l.sum(), num_tokens)
		if (epoch + 1) % 10 == 0:
			animator.add(epoch + 1, (metric[0] / metric[1],))
	print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} ' 
		f'tokens/sec on {str(device)}')

if __name__ == '__main__':
	# 以创建和训练⼀个循环神经⽹络“编码器－解码器”模型⽤于序列到序列的学习。
	embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
	batch_size, num_steps = 64, 10
	lr, num_epochs, device = 0.005, 300, train_framework.try_gpu()
	train_iter, src_vocab, tgt_vocab = \
		test_machine_translation_and_dataset.load_data_nmt(batch_size, num_steps)
	encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
					dropout)
	decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
					dropout)
	net = my_encoder_decoder.EncoderDecoder(encoder, decoder)
	train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
				device, save_attention_weights=False):
	"""序列到序列模型的预测"""
	# 在预测时将net设置为评估模式
	net.eval()
	# 处理测试用数据。全部转为小写以后，进行分词，之后添加<eos>结尾。
	src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
				src_vocab['<eos>']]
	# 获得单词个数。
	enc_valid_len = torch.tensor([len(src_tokens)], device=device)
	# 截断或填充⽂本序列。
	src_tokens = test_machine_translation_and_dataset.truncate_pad(
			src_tokens, num_steps, src_vocab['<pad>'])
	# 添加批量轴
	enc_X = torch.unsqueeze(
				torch.tensor(src_tokens, 
					dtype=torch.long, device=device), dim=0)
	# 调用编码器编码。
	enc_outputs = net.encoder(enc_X, enc_valid_len)
	# 初始化解码器。
	dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
	# 添加批量轴
	dec_X = torch.unsqueeze(torch.tensor(
		[tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
	output_seq, attention_weight_seq = [], []
	# 开始预测。
	for _ in range(num_steps):
		# 使用初始化状态调用解码器，并更新状态。
		Y, dec_state = net.decoder(dec_X, dec_state)
		# 我们使⽤具有预测最⾼可能性的词元，作为解码器在下⼀时间步的输⼊
		dec_X = Y.argmax(dim=2)
		pred = dec_X.squeeze(dim=0).type(torch.int32).item()
		# 保存注意⼒权重（稍后讨论）
		if save_attention_weights:
			attention_weight_seq.append(net.decoder.attention_weights)
		# ⼀旦序列结束词元被预测，输出序列的⽣成就完成了
		if pred == tgt_vocab['<eos>']:
			break
		output_seq.append(pred)
	return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
# BLEU（bilingual evaluation understudy）
# 已经被⼴泛⽤于测量许多应⽤的输出序列的质量。
# 这里面提到的pn的含义如下：
# 例如，给定标签序列A、B、C、D、E、F 和预测序列A、B、B、C、D，
# 我们有p1 = 4/5、p2 = 3/4、p3 = 1/3和p4 = 0。
# 也就是对于标签数据来说，预测序列出现了其中的4个字母ABCD。而预测序列则有5个元素。则p1 = 4/5
# 而对于标签数据来说，2个字母的序列，在预测序列出现了3次。分别是AB，BC，CD。
# 而预测数据中，2个字母的序列，出现了4次。分别是AB，BB，BC，CD。则p2 = 3/4
# 同理，对于标签数据来说，3个字母的序列，在预测序列出现了1次。分别是BCD。
# 而预测数据中，3个字母的序列，出现了3次。分别是ABB，BBC，BCD。则p3 = 1/3
# 同理，对于标签数据来说，4个字母的序列，在预测序列没有出现。则p4 = 0
def bleu(pred_seq, label_seq, k): #@save
	"""计算BLEU"""
	# 取得数据和标签。
	pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
	# 取得数据和标签的长度。
	len_pred, len_label = len(pred_tokens), len(label_tokens)
	# 使用BLEU公式计算惩罚因子。
	score = math.exp(min(0, 1 - len_label / len_pred))
	for n in range(1, k + 1):
		num_matches, label_subs = 0, collections.defaultdict(int)
		# 下面就是根据上面的例子计算p1...pn
		for i in range(len_label - n + 1):
			label_subs[' '.join(label_tokens[i: i + n])] += 1
		for i in range(len_pred - n + 1):
			if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
				num_matches += 1
			label_subs[' '.join(pred_tokens[i: i + n])] -= 1
		# 计算pn^(1/2^n)
		score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
	return score
	
if __name__ == '__main__':
	# 最后，利⽤训练好的循环神经⽹络“编码器－解码器”模型，
	# 将⼏个英语句⼦翻译成法语，并计算BLEU的最终结果。
	engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
	fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
	for eng, fra in zip(engs, fras):
		translation, attention_weight_seq = predict_seq2seq(
				net, eng, src_vocab, tgt_vocab, num_steps, device)
		print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

