import os
import re
import torch
from torch import nn

import my_download
import train_framework
import my_data_time_machine
import test_machine_translation_and_dataset

#@save
my_download.DATA_HUB['SNLI'] = ( 
	'https://nlp.stanford.edu/projects/snli/snli_1.0.zip', 
	'9fcde07509c7e87ec61c640c1b2753d9041758e4')
data_dir = my_download.download_extract('SNLI')

# 原始的SNLI数据集包含的信息⽐我们在实验中真正需要的信息丰富得多。
# 因此，我们定义函数read_snli以仅提取数据集的⼀部分，然后返回前提、假设及其标签的列表。
#@save
def read_snli(data_dir, is_train):
	"""将SNLI数据集解析为前提、假设和标签"""
	def extract_text(s):
		# 删除我们不会使⽤的信息
		s = re.sub('\\(', '', s)
		s = re.sub('\\)', '', s)
		# ⽤⼀个空格替换两个或多个连续的空格
		s = re.sub('\\s{2,}', ' ', s)
		return s.strip()
	label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
	file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
			if is_train else 'snli_1.0_test.txt')
	with open(file_name, 'r') as f:
		rows = [row.split('\t') for row in f.readlines()[1:]]
	premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
	hypotheses = [extract_text(row[2]) for row in rows if row[0] \
				in label_set]
	labels = [label_set[row[0]] for row in rows if row[0] in label_set]
	return premises, hypotheses, labels

if __name__ == '__main__':
	# 现在让我们打印前3对前提和假设，以及它们的标签（“0”、“1”和“2”分别对应于“蕴涵”、“⽭盾”和“中性”）。
	train_data = read_snli(data_dir, is_train=True)
	for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
		print('前提：', x0)
		print('假设：', x1)
		print('标签：', y)

	# 训练集约有550000对，测试集约有10000对。
	# 下⾯显⽰了训练集和测试集中的三个标签“蕴涵”、“⽭盾”和“中性”是平衡的。
	test_data = read_snli(data_dir, is_train=False)
	for data in [train_data, test_data]:
		print([[row for row in data[2]].count(i) for i in range(3)])

# 下⾯我们来定义⼀个⽤于加载SNLI数据集的类。
#@save
class SNLIDataset(torch.utils.data.Dataset):
	"""⽤于加载SNLI数据集的⾃定义数据集"""
	# 类构造函数中的变量num_steps指定⽂本序列的⻓度，使得每个⼩批量序列将具有相同的形状。
	def __init__(self, dataset, num_steps, vocab=None):
		self.num_steps = num_steps
		all_premise_tokens = my_data_time_machine.tokenize(dataset[0])
		all_hypothesis_tokens = my_data_time_machine.tokenize(dataset[1])
		if vocab is None:
			self.vocab = my_data_time_machine.Vocab(all_premise_tokens + \
				all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
		else:
			self.vocab = vocab
		self.premises = self._pad(all_premise_tokens)
		self.hypotheses = self._pad(all_hypothesis_tokens)
		self.labels = torch.tensor(dataset[2])
		print('read ' + str(len(self.premises)) + ' examples')
	def _pad(self, lines):
		# 换句话说，在较⻓序列中的前num_steps个标记之后的标记被截断，
		# ⽽特殊标记“<pad>”将被附加到较短的序列后，直到它们的⻓度变为num_steps。
		return torch.tensor([test_machine_translation_and_dataset.truncate_pad(
			self.vocab[line], self.num_steps, self.vocab['<pad>'])
				for line in lines])
	# 通过实现__getitem__功能，我们可以任意访问带有索引idx的前提、假设和标签。
	def __getitem__(self, idx):
		return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]
	def __len__(self):
		return len(self.premises)

# 注意，我们必须使⽤从训练集构造的词表作为测试集的词表。
# 因此，在训练集中训练的模型将不知道来⾃测试集的任何新词元。
#@save
def load_data_snli(batch_size, num_steps=50):
	"""下载SNLI数据集并返回数据迭代器和词表"""
	# num_workers = train_framework.get_dataloader_workers()
	num_workers = 0
	data_dir = my_download.download_extract('SNLI')
	# 现在，我们可以调⽤read_snli函数和SNLIDataset类来下载SNLI数据集，
	train_data = read_snli(data_dir, True)
	test_data = read_snli(data_dir, False)
	train_set = SNLIDataset(train_data, num_steps)
	test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
	train_iter = torch.utils.data.DataLoader(train_set, batch_size,
							shuffle=True,
							num_workers=num_workers)
	test_iter = torch.utils.data.DataLoader(test_set, batch_size,
							shuffle=False,
							num_workers=num_workers)
	# 并返回训练集和测试集的DataLoader实例，以及训练集的词表。
	return train_iter, test_iter, train_set.vocab

if __name__ == '__main__':
	# 在这⾥，我们将批量⼤⼩设置为128时，将序列⻓度设置为50，
	# 并调⽤load_data_snli函数来获取数据迭代器和词表。然后我们打印词表⼤⼩。
	train_iter, test_iter, vocab = load_data_snli(128, 50)
	print("len(vocab) : ", len(vocab))
	# 现在我们打印第⼀个⼩批量的形状。与情感分析相反，我们有分别代表前提和假设的两个输⼊X[0]和X[1]。
	for X, Y in train_iter:
		print("X[0].shape : ", X[0].shape)
		print("X[1].shape : ", X[1].shape)
		print("Y.shape    : ", Y.shape)
		break

