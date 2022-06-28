import random
import torch
import test_text_preprocessing
import my_plt

if __name__ == '__main__':
	# 根据 8.2节中介绍的时光机器数据集构建词表，并打印前10个最常⽤的（频率最⾼的）单词。
	tokens = test_text_preprocessing.tokenize(
		test_text_preprocessing.read_time_machine())
	# 因为每个⽂本⾏不⼀定是⼀个句⼦或⼀个段落，因此我们把所有⽂本⾏拼接到⼀起
	corpus = [token for line in tokens for token in line]
	vocab = test_text_preprocessing.Vocab(corpus)
	print("vocab.token_freqs[:10] : ", vocab.token_freqs[:10])

	# 通过此图我们可以发现：词频以⼀种明确的⽅式迅速衰减。
	# 将前⼏个单词作为例外消除后，剩余的所有单词⼤致遵循双对数坐标图上的⼀条直线。
	# 这意味着单词的频率满⾜⻬普夫定律（Zipf’s law）
	freqs = [freq for token, freq in vocab.token_freqs]
	my_plt.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
	 		xscale='log', yscale='log')

	# 我们来看看⼆元语法的频率是否与⼀元语法的频率表现出相同的⾏为⽅式。
	bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
	bigram_vocab = test_text_preprocessing.Vocab(bigram_tokens)
	# 在⼗个最频繁的词对中，有九个是由两个停⽤词组成的，只有⼀个与“the time”有关。
	print("bigram_vocab.token_freqs[:10] : ", bigram_vocab.token_freqs[:10])
	# 再进⼀步看看三元语法的频率是否表现出相同的⾏为⽅式。
	trigram_tokens = [triple for triple in zip(
					corpus[:-2], corpus[1:-1], corpus[2:])]
	trigram_vocab = test_text_preprocessing.Vocab(trigram_tokens)
	print("trigram_vocab.token_freqs[:10] : ", trigram_vocab.token_freqs[:10])
	# 最后，我们直观地对⽐三种模型中的词元频率：⼀元语法、⼆元语法和三元语法。
	bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
	trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
	my_plt.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
			ylabel='frequency: n(x)', xscale='log', yscale='log',
			legend=['unigram', 'bigram', 'trigram'])

# 下⾯的代码每次可以从数据中随机⽣成⼀个⼩批量。
# 参数batch_size指定了每个⼩批量中⼦序列样本的数⽬，
# 参数num_steps是每个⼦序列中预定义的时间步数。
def seq_data_iter_random(corpus, batch_size, num_steps): #@save
	"""使⽤随机抽样⽣成⼀个⼩批量⼦序列"""
	# 从随机偏移量开始对序列进⾏分区，随机范围包括num_steps-1
	corpus = corpus[random.randint(0, num_steps - 1):]
	# 减去1，是因为我们需要考虑标签
	num_subseqs = (len(corpus) - 1) // num_steps
	# ⻓度为num_steps的⼦序列的起始索引
	initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
	# 在随机抽样的迭代过程中，
	# 来⾃两个相邻的、随机的、⼩批量中的⼦序列不⼀定在原始序列上相邻
	random.shuffle(initial_indices)

	def data(pos):
		# 返回从pos位置开始的⻓度为num_steps的序列
		return corpus[pos: pos + num_steps]
	# 从数据中随机⽣成⼀个⼩批量。
	num_batches = num_subseqs // batch_size
	for i in range(0, batch_size * num_batches, batch_size):
		# 在这⾥，initial_indices包含⼦序列的随机起始索引
		initial_indices_per_batch = initial_indices[i: i + batch_size]
		X = [data(j) for j in initial_indices_per_batch]
		Y = [data(j + 1) for j in initial_indices_per_batch]
		yield torch.tensor(X), torch.tensor(Y)

if __name__ == '__main__':
	# 下⾯我们⽣成⼀个从0到34的序列。假设批量⼤⼩为2，时间步数为5，
	# 这意味着可以⽣成 ⌊(35 5 1)/5⌋ = 6个“特征－标签”⼦序列对。
	# 如果设置⼩批量⼤⼩为2，我们只能得到3个⼩批量。
	my_seq = list(range(35))
	for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
		print('X: ', X, '\nY:', Y)

# 除了对原始序列可以随机抽样外，我们还可以保证两个相邻的⼩批量中的⼦序列在原始序列上也是相邻的。
# 这种策略在基于⼩批量的迭代过程中保留了拆分的⼦序列的顺序，因此称为顺序分区。
def seq_data_iter_sequential(corpus, batch_size, num_steps): #@save
	"""使⽤顺序分区⽣成⼀个⼩批量⼦序列"""
	# 从随机偏移量开始划分序列
	offset = random.randint(0, num_steps)
	# num_tokens需要是batch_size的倍数。
	num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
	# 从随机偏移量开始。取batch_size的倍数的最大长度。
	Xs = torch.tensor(corpus[offset: offset + num_tokens])
	Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
	# 展平数据。
	Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
	# 获得批量个数
	num_batches = Xs.shape[1] // num_steps
	# 从数据中随机⽣成⼀个⼩批量。
	for i in range(0, num_steps * num_batches, num_steps):
		X = Xs[:, i: i + num_steps]
		Y = Ys[:, i: i + num_steps]
		yield X, Y

if __name__ == '__main__':
	# 基于相同的设置，通过顺序分区读取每个⼩批量的⼦序列的特征X和标签Y。
	# 通过将它们打印出来可以发现：
	# 	 迭代期间来⾃两个相邻的⼩批量中的⼦序列在原始序列中确实是相邻的。
	for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
		print('X: ', X, '\nY:', Y)

# 现在，我们将上⾯的两个采样函数包装到⼀个类中，以便稍后可以将其⽤作数据迭代器。
class SeqDataLoader: #@save
	"""加载序列数据的迭代器"""
	def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
		if use_random_iter:
			self.data_iter_fn = seq_data_iter_random
		else:
			self.data_iter_fn = seq_data_iter_sequential
		self.corpus, self.vocab = test_text_preprocessing.load_corpus_time_machine(max_tokens)
		self.batch_size, self.num_steps = batch_size, num_steps
	def __iter__(self):
		return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

# 定义了⼀个函数load_data_time_machine，它同时返回数据迭代器和词表。
def load_data_time_machine(batch_size, num_steps, #@save
				use_random_iter=False, max_tokens=10000):
	"""返回时光机器数据集的迭代器和词表"""
	data_iter = SeqDataLoader(
			batch_size, num_steps, use_random_iter, max_tokens)
	return data_iter, data_iter.vocab


