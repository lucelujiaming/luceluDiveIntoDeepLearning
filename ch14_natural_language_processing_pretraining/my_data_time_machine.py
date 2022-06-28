import torch
from torch import nn
import random
import collections
import re
import my_download

#@save
my_download.DATA_HUB['time_machine'] = (my_download.DATA_URL + 'timemachine.txt', 
	'090b5e7e70c295757f55df93cb0a180b9691891a')
# 读取数据集
def read_time_machine(): #@save
	"""将时间机器数据集加载到⽂本⾏的列表中"""
	# 将数据集读取到由多条⽂本⾏组成的列表中，其中每条⽂本⾏都是⼀个字符串。
	with open(my_download.download('time_machine'), 'r') as f:
		lines = f.readlines()
	# 为简单起⻅，我们在这⾥忽略了标点符号和字⺟⼤写。
	return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
# 下⾯的tokenize函数将⽂本⾏列表（lines）作为输⼊，
# 列表中的每个元素是⼀个⽂本序列（如⼀条⽂本⾏）。
def tokenize(lines, token='word'): #@save
	"""将⽂本⾏拆分为单词或字符词元"""
	if token == 'word':
		# 每个⽂本序列⼜被拆分成⼀个词元列表，词元（token）是⽂本的基本单位。
		# 最后，返回⼀个由词元列表组成的列表，其中的每个词元都是⼀个字符串（string）。
		return [line.split() for line in lines]
	elif token == 'char':
		return [list(line) for line in lines]
	else:
		print('错误：未知词元类型：' + token)

# 将字符串类型的词元映射到从0开始的数字索引中。
# 词元的类型是字符串，⽽模型需要的输⼊是数字，因此这种类型不⽅便模型使⽤。
# 我们需要构建⼀个字典，通常也叫做词表（vocabulary），
# ⽤来将字符串类型的词元映射到从0开始的数字索引中。
class Vocab: #@save
	"""⽂本词表"""
	def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
		if tokens is None:
			tokens = []
		if reserved_tokens is None:
			reserved_tokens = []
		# 按出现频率排序。
		counter = count_corpus(tokens)
		self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
									reverse=True) 
		# 未知词元的索引为0
		# 初始化idx_to_token，初始化为：['<unk>']
		self.idx_to_token = ['<unk>'] + reserved_tokens
		# 初始化token_to_idx，初始化为：{'<unk>': 0}
		self.token_to_idx = {token: idx
					for idx, token in enumerate(self.idx_to_token)}
		# 循环排序好的词表。
		for token, freq in self._token_freqs:
			# 如果发现出现频率低于最低频率，就退出循环。
			if freq < min_freq:
				break
			# 如果高于某个频率。
			if token not in self.token_to_idx:
				# 就把这个token追加到idx_to_token中。
				self.idx_to_token.append(token)
				# token_to_idx保存这个token和对应的数字索引。
				self.token_to_idx[token] = len(self.idx_to_token) - 1
	# 返回词元个数。
	def __len__(self):
		return len(self.idx_to_token)
	# 获得词元对应的token_to_idx元素对象。
	def __getitem__(self, tokens):
		if not isinstance(tokens, (list, tuple)):
			return self.token_to_idx.get(tokens, self.unk)
		return [self.__getitem__(token) for token in tokens]
	def to_tokens(self, indices):
		if not isinstance(indices, (list, tuple)):
			return self.idx_to_token[indices]
		return [self.idx_to_token[index] for index in indices]
	@property
	def unk(self): # 未知词元的索引为0
		return 0
	@property
	def token_freqs(self):
		return self._token_freqs
def count_corpus(tokens): #@save
	"""统计词元的频率"""
	# 这⾥的tokens是1D列表或2D列表
	if len(tokens) == 0 or isinstance(tokens[0], list):
		# 将词元列表展平成⼀个列表
		tokens = [token for line in tokens for token in line]
	# collections.Counter计数来体现。
	# 该方法用于统计某序列中每个元素出现的次数，以键值对的方式存在字典中。
	return collections.Counter(tokens)
		
# from test_language_models_and_dataset.py
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
# 我们将所有功能打包到load_corpus_time_machine函数中，
# 该函数返回corpus（词元索引列表）和vocab（时光机器语料库的词表）。我们在这⾥所做的改变是：
#   1. 为了简化后⾯章节中的训练，我们使⽤字符（⽽不是单词）实现⽂本词元化；
#   2. 时光机器数据集中的每个⽂本⾏不⼀定是⼀个句⼦或⼀个段落，
#      还可能是⼀个单词，因此返回的corpus仅处理为单个列表，⽽不是使⽤多词元列表构成的⼀个列表。
def load_corpus_time_machine(max_tokens=-1): #@save
	"""返回时光机器数据集的词元索引列表和词表"""
	# 读取数据集
	lines = read_time_machine()
	# 将⽂本⾏列表（lines）作为输⼊，将⽂本⾏拆分为单词或字符词元。
	tokens = tokenize(lines, 'char')
	# 将字符串类型的词元映射到从0开始的数字索引中。
	vocab = Vocab(tokens)
	# 因为时光机器数据集中的每个⽂本⾏不⼀定是⼀个句⼦或⼀个段落，
	# 所以将所有⽂本⾏展平到⼀个列表中。
	corpus = [vocab[token] for line in tokens for token in line]
	if max_tokens > 0:
		corpus = corpus[:max_tokens]
	return corpus, vocab
# 现在，我们将上⾯的两个采样函数包装到⼀个类中，以便稍后可以将其⽤作数据迭代器。
class SeqDataLoader: #@save
	"""加载序列数据的迭代器"""
	def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
		if use_random_iter:
			self.data_iter_fn = seq_data_iter_random
		else:
			self.data_iter_fn = seq_data_iter_sequential
		self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
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