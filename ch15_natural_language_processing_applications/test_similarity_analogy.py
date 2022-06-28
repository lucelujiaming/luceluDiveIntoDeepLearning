import os
import torch
from torch import nn

import my_timer
import my_download
import train_framework
import test_text_preprocessing
import test_machine_translation_and_dataset

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 以下列出维度为50、100和300的预训练GloVe嵌⼊，可从GloVe⽹站196下载。
# 预训练的fastText嵌⼊有多种语⾔。
# 这⾥我们使⽤可以从fastText⽹站197下载300维度的英⽂版本（“wiki.en”）。
#@save
my_download.DATA_HUB['glove.6b.50d'] = (my_download.DATA_URL + 'glove.6B.50d.zip', 
	'0b8703943ccdb6eb788e6f091b8946e82231bc4d')
#@save
my_download.DATA_HUB['glove.6b.100d'] = (my_download.DATA_URL + 'glove.6B.100d.zip', 
	'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')
#@save
my_download.DATA_HUB['glove.42b.300d'] = (my_download.DATA_URL + 'glove.42B.300d.zip', 
	'b5116e234e9eb9076672cfeabf5469f3eec904fa')
#@save
my_download.DATA_HUB['wiki.en'] = (my_download.DATA_URL + 'wiki.en.zip', 
	'c1816da3821ae9f43899be655002f6c723e91b88')

# 为了加载这些预训练的GloVe和fastText嵌⼊，我们定义了以下TokenEmbedding类。
#@save
class TokenEmbedding:
	"""GloVe嵌⼊"""
	def __init__(self, embedding_name):
		self.idx_to_token, self.idx_to_vec = self._load_embedding(
							embedding_name)
		self.unknown_idx = 0
		self.token_to_idx = {token: idx for idx, token in
							enumerate(self.idx_to_token)}
	def _load_embedding(self, embedding_name):
		idx_to_token, idx_to_vec = ['<unk>'], []
		data_dir = my_download.download_extract(embedding_name)
		# GloVe⽹站：https://nlp.stanford.edu/projects/glove/
		# fastText⽹站：https://fasttext.cc/
		with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
			for line in f:
				elems = line.rstrip().split(' ')
				token, elems = elems[0], [float(elem) for elem in elems[1:]]
				# 跳过标题信息，例如fastText中的⾸⾏
				if len(elems) > 1:
					idx_to_token.append(token)
					idx_to_vec.append(elems)
		idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
		return idx_to_token, torch.tensor(idx_to_vec)
	def __getitem__(self, tokens):
		indices = [self.token_to_idx.get(token, self.unknown_idx)
						for token in tokens]
		vecs = self.idx_to_vec[torch.tensor(indices)]
		return vecs
	def __len__(self):
		return len(self.idx_to_token)
		
if __name__ == '__main__':
	# 下⾯我们加载50维GloVe嵌⼊（在维基百科的⼦集上预训练）。
	# 创建TokenEmbedding实例时，如果尚未下载指定的嵌⼊⽂件，则必须下载该⽂件。
	glove_6b50d = TokenEmbedding('glove.6b.50d')
	# 输出词表⼤⼩。
	print("len(glove_6b50d) : ", len(glove_6b50d))
	# 我们可以得到词表中⼀个单词的索引，反之亦然。
	print("glove_6b50d.token_to_idx['beautiful'] : ", 
		glove_6b50d.token_to_idx['beautiful'])
	print("glove_6b50d.idx_to_token[3367] : ",
		glove_6b50d.idx_to_token[3367])
	# glove_6b100d = TokenEmbedding('glove.6b.100d')
	# print("len(glove_6b100d) : ", len(glove_6b100d))
	# glove_42b300d = TokenEmbedding('glove.42b.300d')
	# print("len(glove_42b300d) : ", len(glove_42b300d))
	# glove_WikiEn = TokenEmbedding('wiki.en')
	# print("len(glove_WikiEn) : ", len(glove_WikiEn))

# 为了根据词向量之间的余弦相似性为输⼊词查找语义相似的词，
# 我们实现了以下knn（k近邻）函数。
def knn(W, x, k):
	# 增加1e-9以获得数值稳定性
	cos = torch.mv(W, x.reshape(-1,)) / (
		torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
		torch.sqrt((x * x).sum()))
	_, topk = torch.topk(cos, k=k)
	return topk, [cos[int(i)] for i in topk]
# 然后，我们使⽤TokenEmbedding的实例embed中预训练好的词向量来搜索相似的词。
def get_similar_tokens(query_token, k, embed):
	topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
	for i, c in zip(topk[1:], cos[1:]): # 排除输⼊词
		print(f'{embed.idx_to_token[int(i)]}：cosine相似度={float(c):.3f}')
if __name__ == '__main__':
	# 我们在词表中找到与“chip”⼀词语义最相似的三个词。
	get_similar_tokens('chip', 3, glove_6b50d)

	# 下⾯输出与“baby”和“beautiful”相似的词。
	get_similar_tokens('baby', 3, glove_6b50d)
	get_similar_tokens('beautiful', 3, glove_6b50d)

# 除了找到相似的词，我们还可以将词向量应⽤到词类⽐任务中。
# 具体来说，词类⽐任务可以定义为：对于单词类⽐a : b :: c : d，
def get_analogy(token_a, token_b, token_c, embed):
	# 给出前三个词a、b和c，找到d。⽤vec(w)表⽰词w的向量，
	vecs = embed[[token_a, token_b, token_c]]
	x = vecs[1] - vecs[0] + vecs[2]
	# 为了完成这个类⽐，我们将找到⼀个词，其向量与vec(c) + vec(b) − vec(a)的结果最相似。
	topk, cos = knn(embed.idx_to_vec, x, 1)
	return embed.idx_to_token[int(topk[0])] # 删除未知词
if __name__ == '__main__':
	# 让我们使⽤加载的词向量来验证“male-female”类⽐。
	retAnalogy = get_analogy('man', 'woman', 'son', glove_6b50d)
	print("retAnalogy : ", retAnalogy)

	# 下⾯完成⼀个“⾸都-国家”的类⽐：“beijing”: “china”:: “tokyo”: “japan”。
	# 这说明了预训练词向量中的语义。
	retAnalogy = get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
	print("retAnalogy : ", retAnalogy)
	# 另外，对于“bad”: “worst”:: “big”: “biggest”等“形容词-形容词最⾼级”的⽐喻，
	# 预训练词向量可以捕捉到句法信息。
	retAnalogy = get_analogy('bad', 'worst', 'big', glove_6b50d)
	print("retAnalogy : ", retAnalogy)
	# 为了演⽰在预训练词向量中捕捉到的过去式概念，
	# 我们可以使⽤“现在式-过去式”的类⽐来测试句法：“do” : “did”:: “go”: “went”。
	retAnalogy = get_analogy('do', 'did', 'go', glove_6b50d)
	print("retAnalogy : ", retAnalogy)

