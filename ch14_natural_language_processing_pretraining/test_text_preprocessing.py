import collections
import re
import train_framework
import my_download

# ⽂本的常⻅预处理步骤通常包括：
#   1. 将⽂本作为字符串加载到内存中。
#   2. 将字符串拆分为词元（如单词和字符）。
#   3. 建⽴⼀个词表，将拆分的词元映射到数字索引。
#   4. 将⽂本转换为数字索引序列，⽅便模型操作。

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

if __name__ == '__main__':
	lines = read_time_machine()
	print(f'# ⽂本总⾏数: {len(lines)}')
	print("lines[0]  : ", lines[0])
	print("lines[10] : ", lines[10])

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

if __name__ == '__main__':
	tokens = tokenize(lines)
	for i in range(11):
		print("tokenize return tokens[", i, "] : ", tokens[i])

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

if __name__ == '__main__':
	# 我们⾸先使⽤时光机器数据集作为语料库来构建词表，然后打印前⼏个⾼频词元及其索引。
	vocab = Vocab(tokens)
	print("vocab.token_to_idx.items()[:10] : ", list(vocab.token_to_idx.items())[:10])
	# 现在，我们可以将每⼀条⽂本⾏转换成⼀个数字索引列表。
	for i in [0, 10]:
		print('⽂本(', i, '):', tokens[i])
		print('索引(', i, '):', vocab[tokens[i]])

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

if __name__ == '__main__':
	corpus, vocab = load_corpus_time_machine()
	len(corpus), len(vocab)





