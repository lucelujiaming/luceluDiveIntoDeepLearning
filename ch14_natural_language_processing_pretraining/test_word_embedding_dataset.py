import math
import os
import random
import torch

import my_timer
import my_download
import train_framework
import test_text_preprocessing
import test_machine_translation_and_dataset

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 我们在这⾥使⽤的数据集是Penn Tree Bank（PTB）191。
# 该语料库取⾃“华尔街⽇报”的⽂章，分为训练集、验证集和测试集。
#@save
my_download.DATA_HUB['ptb'] = (my_download.DATA_URL + 'ptb.zip',
			 '319d85e578af0cdc590547f26231e4e31cdf1e42')
# 在原始格式中，⽂本⽂件的每⼀⾏表⽰由空格分隔的⼀句话。在这⾥，
# 我们将每个单词视为⼀个词元。
#@save
def read_ptb():
	"""将PTB数据集加载到⽂本⾏的列表中"""
	data_dir = my_download.download_extract('ptb')
	# Readthetrainingset.
	with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
		raw_text = f.read()
	return [line.split() for line in raw_text.split('\n')]
if __name__ == '__main__':
	sentences = read_ptb()

	# 在读取训练集之后，我们为语料库构建了⼀个词表，
	# 其中出现次数少于10次的任何单词都将由“<unk>”词元替换。
	# 请注意，原始数据集还包含表⽰稀有（未知）单词的“<unk>”词元。
	vocab = test_text_preprocessing.Vocab(sentences, min_freq=10) 
	print(f'vocab size: {len(vocab)}')

# ⽂本数据通常有“the”、“a”和“in”等⾼频词：它们在⾮常⼤的语料库中甚⾄可能出现数⼗亿次。
# 当训练词嵌⼊模型时，可以对⾼频单词进⾏下采样 [Mikolov et al., 2013b]。
# 具体地说，数据集中的每个词wi将有概率地被丢弃。
# 只有当相对⽐率f(wi) > t时，（⾼频）词wi才能被丢弃，且该词的相对⽐率越⾼，被丢弃的概率就越⼤。
#@save
def subsample(sentences, vocab):
	"""下采样⾼频词"""
	# 排除未知词元'<unk>'
	sentences = [[token for token in line if vocab[token] != vocab.unk]
						for line in sentences]
	counter = test_text_preprocessing.count_corpus(sentences)
	num_tokens = sum(counter.values())
	# 如果在下采样期间保留词元，则返回True
	def keep(token):
		# 参见(14.3.1)。
		return(random.uniform(0, 1) <
			math.sqrt(1e-4 / counter[token] * num_tokens))
	return ([[token for token in line if keep(token)] for line in sentences],
		counter)
if __name__ == '__main__':
	subsampled, counter = subsample(sentences, vocab)
	# 下⾯的代码⽚段绘制了下采样前后每句话的词元数量的直⽅图。
	# 正如预期的那样，下采样通过删除⾼频词来显著缩短句⼦，这将使训练加速。
	# test_machine_translation_and_dataset.show_list_len_pair_hist(
	# 		['origin', 'subsampled'], '# tokens per sentence', 'count', 
	# 		 sentences, subsampled);

# 对于单个词元，⾼频词“the”的采样率不到1/20。
def compare_counts(token):
	return (f'"{token}"的数量：' 
		f'之前={sum([l.count(token) for l in sentences])}, ' 
		f'之后={sum([l.count(token) for l in subsampled])}')
if __name__ == '__main__':
	# 对于单个词元，⾼频词“the”的采样率不到1/20。
	print(compare_counts('the'))
	print(compare_counts('join'))
	# 在下采样之后，我们将词元映射到它们在语料库中的索引。
	corpus = [vocab[line] for line in subsampled]
	print("corpus[:3] : ", corpus[:3])

# 下⾯的get_centers_and_contexts函数从corpus中提取所有中⼼词及其上下⽂词。
#@save
def get_centers_and_contexts(corpus, max_window_size):
	"""返回跳元模型中的中⼼词和上下⽂词"""
	centers, contexts = [], []
	for line in corpus:
		# 要形成“中⼼词-上下⽂词”对，每个句⼦⾄少需要有2个词
		if len(line) < 2:
			continue
		centers += line
		for i in range(len(line)): # 上下⽂窗⼝中间i
			# 它随机采样1到max_window_size之间的整数作为上下⽂窗⼝。
			window_size = random.randint(1, max_window_size)
			# 对于任⼀中⼼词，与其距离不超过采样上下⽂窗⼝⼤⼩的词为其上下⽂词。
			indices = list(range(max(0, i - window_size),
								 min(len(line), i + 1 + window_size)))
			# 从上下⽂词中排除中⼼词
			indices.remove(i)
			contexts.append([line[idx] for idx in indices])
	return centers, contexts

if __name__ == '__main__':
	# 接下来，我们创建⼀个⼈⼯数据集，分别包含7个和3个单词的两个句⼦。
	tiny_dataset = [list(range(7)), list(range(7, 10))]
	print('数据集', tiny_dataset)
	# 设置最⼤上下⽂窗⼝⼤⼩为2，并打印所有中⼼词及其上下⽂词。
	for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
		print('中⼼词', center, '的上下⽂词是', context)

	# 在PTB数据集上进⾏训练时，我们将最⼤上下⽂窗⼝⼤⼩设置为5。
	# 下⾯提取数据集中的所有中⼼词及其上下⽂词。
	all_centers, all_contexts = get_centers_and_contexts(corpus, 5) 
	print(f'# “中⼼词-上下⽂词对”的数量: {sum([len(contexts) for contexts in all_contexts])}')

# 我们使⽤负采样进⾏近似训练。为了根据预定义的分布对噪声词进⾏采样，
# 我们定义以下RandomGenerator类。
#@save
class RandomGenerator:
	"""根据n个采样权重在{1,...,n}中随机抽取"""
	def __init__(self, sampling_weights):
		# Exclude
		self.population = list(range(1, len(sampling_weights) + 1))
		# （可能未规范化的）采样分布通过变量sampling_weights传递。
		self.sampling_weights = sampling_weights
		self.candidates = []
		self.i = 0
	def draw(self):
		if self.i == len(self.candidates):
			# 缓存k个随机采样结果
			self.candidates = random.choices(
				self.population, self.sampling_weights, k=10000)
			self.i = 0
		self.i += 1
		return self.candidates[self.i - 1]

if __name__ == '__main__':
	# 我们可以在索引1、2和3中绘制10个随机变量X
	generator = RandomGenerator([2, 3, 4])
	print("在索引1、2和3中绘制10个随机变量X : ", 
		[generator.draw() for _ in range(10)])

# 对于⼀对中⼼词和上下⽂词，我们随机抽取了K个（实验中为5个）噪声词。
def get_negatives(all_contexts, vocab, counter, K):
	"""返回负采样中的噪声词"""
	# 索引为1、2、...（索引0是词表中排除的未知标记）
	# 根据word2vec论⽂中的建议，将噪声词w的采样概率P(w)设置为其在字典中的相对频率，其幂为0.75
	sampling_weights = [counter[vocab.to_tokens(i)]**0.75
					for i in range(1, len(vocab))]
	# 根据word2vec论⽂中的建议，
	# 将噪声词w的采样概率P(w)设置为其在字典中的相对频率。
	all_negatives, generator = [], RandomGenerator(sampling_weights)
	for contexts in all_contexts:
		negatives = []
		while len(negatives) < len(contexts) * K:
			neg = generator.draw()
			# 噪声词不能是上下⽂词
			if neg not in contexts:
				negatives.append(neg)
		all_negatives.append(negatives)
	return all_negatives
if __name__ == '__main__':
	all_negatives = get_negatives(all_contexts, vocab, counter, 5)

# 在提取所有中⼼词及其上下⽂词和采样噪声词后，将它们转换成⼩批量的样本，在训练过程中可以迭代加载。
# 上述思想在下⾯的batchify函数中实现。其输⼊data是⻓度等于批量⼤⼩的列表，
# 其中每个元素是由中⼼词center、其上下⽂词context和其噪声词negative组成的样本。
# 此函数返回⼀个可以在训练期间加载⽤于计算的⼩批量，例如包括掩码变量。
# 输入参数：
#     data - ⻓度等于批量⼤⼩的列表，
#            其中每个元素是由中⼼词center、其上下⽂词context和其噪声词negative组成的样本。
#@save
def batchify(data):
	"""返回带有负采样的跳元模型的⼩批量样本"""
	max_len = max(len(c) + len(n) for _, c, n in data)
	centers, contexts_negatives, masks, labels = [], [], [], []
	for center, context, negative in data:
		cur_len = len(context) + len(negative)
		centers += [center]
		# 我们在contexts_negatives个变量中将其上下⽂词和噪声词连结起来，
		# 并填充零，直到连结⻓度达到maxi_ni+mi(max_len)。
		contexts_negatives += \
			[context + negative + [0] * (max_len - cur_len)]
		# 为了在计算损失时排除填充，我们定义了掩码变量masks。
		# 在masks中的元素和contexts_negatives中的元素之间存在⼀⼀对应关系，
		# 其中masks中的0（否则为1）对应于contexts_negatives中的填充。
		masks += [[1] * cur_len + [0] * (max_len - cur_len)]
		# 为了区分正反例，我们在contexts_negatives中通过⼀个labels变量将上下⽂词与噪声词分开。
		# 类似于masks，在labels中的元素和contexts_negatives中的元素之间也存在⼀⼀对应关系，
		# 其中labels中 的1（否则为0）对应于contexts_negatives中的上下⽂词的正例。
		labels += [[1] * len(context) + [0] * (max_len - len(context))]
	# 返回⼀个可以在训练期间加载⽤于计算的⼩批量，例如包括掩码变量。
	return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
		contexts_negatives), torch.tensor(masks), torch.tensor(labels))

if __name__ == '__main__':
	# 让我们使⽤⼀个⼩批量的两个样本来测试此函数。
	x_1 = (1, [2, 2], [3, 3, 3, 3])
	x_2 = (1, [2, 2, 2], [3, 3])
	batch = batchify((x_1, x_2))
	names = ['centers', 'contexts_negatives', 'masks', 'labels']
	for name, data in zip(names, batch):
		print(name, '=', data)

# 最后，我们定义了读取PTB数据集并返回数据迭代器和词表的load_data_ptb函数。
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
	"""下载PTB数据集，然后将其加载到内存中"""
	num_workers = train_framework.get_dataloader_workers()
	sentences = read_ptb()
	vocab = test_text_preprocessing.Vocab(sentences, min_freq=10)
	subsampled, counter = subsample(sentences, vocab)
	corpus = [vocab[line] for line in subsampled]
	# 从corpus中提取所有中⼼词及其上下⽂词。
	all_centers, all_contexts = get_centers_and_contexts(
				corpus, max_window_size)
	# 对于⼀对中⼼词和上下⽂词，我们随机抽取num_noise_words个噪声词。
	all_negatives = get_negatives(
				all_contexts, vocab, counter, num_noise_words)
	class PTBDataset(torch.utils.data.Dataset):
		def __init__(self, centers, contexts, negatives):
			assert len(centers) == len(contexts) == len(negatives)
			self.centers = centers
			self.contexts = contexts
			self.negatives = negatives
		def __getitem__(self, index):
			return (self.centers[index], self.contexts[index],
					self.negatives[index])
		def __len__(self):
			return len(self.centers)
	dataset = PTBDataset(all_centers, all_contexts, all_negatives)
	data_iter = torch.utils.data.DataLoader(
		dataset, batch_size, shuffle=True,
		# collate_fn=batchify, num_workers=num_workers)
		# 把 num_workers 改为 0 就可以运行，因为num_workers改为0就不启用多进程，
		# 不会预加载多批次数据进入内存。d2l里默认值是4。
		collate_fn=batchify, num_workers=0)
	return data_iter, vocab

if __name__ == '__main__':
	# 让我们打印数据迭代器的第⼀个⼩批量。
	data_iter, vocab = load_data_ptb(512, 5, 5)
	for batch in data_iter:
		# 这里会调用collate_fn函数指针指向的batchify函数。
		# 函数的参数是一个⻓度等于批量⼤⼩的列表，
		# 其中每个元素是由中⼼词center、其上下⽂词context和其噪声词negative组成的样本。
		# 恰好为PTBDataset的__getitem__函数的返回值。
		# 函数batchify返回⼀个可以在训练期间加载⽤于计算的⼩批量，
		# 包括centers, contexts_negatives, masks, labels。
		for name, data in zip(names, batch):
			print(name, 'shape:', data.shape)
		break

