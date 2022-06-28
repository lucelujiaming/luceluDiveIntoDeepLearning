import os
import torch
from torch import nn

import my_plt
import my_download
import train_framework
import my_data_time_machine
import test_machine_translation_and_dataset

# ⾸先，下载并提取路径../data/aclImdb中的IMDb评论数据集。
#@save
my_download.DATA_HUB['aclImdb'] = ( 
	'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', 
	'01ada507287d82875905620988597833ad4e0903')
data_dir = my_download.download_extract('aclImdb', 'aclImdb')

# 接下来，读取训练和测试数据集。
#@save
def read_imdb(data_dir, is_train):
	"""读取IMDb评论数据集⽂本序列和标签"""
	data, labels = [], []
	for label in ('pos', 'neg'):
		folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
					label)
		for file in os.listdir(folder_name):
			with open(os.path.join(folder_name, file), 'rb') as f:
				review = f.read().decode('utf-8').replace('\n', '')
				data.append(review)
				# 每个样本都是⼀个评论及其标签：1表⽰“积极”，0表⽰“消极”。
				labels.append(1 if label == 'pos' else 0)
	return data, labels
if __name__ == '__main__':
	train_data = read_imdb(data_dir, is_train=True)
	print('训练集数⽬：', len(train_data[0]))
	print('标签：', y, 'review:', x[0:60])

	# 将每个单词作为⼀个词元，过滤掉出现不到5次的单词，我们从训练数据集中创建⼀个词表。
	train_tokens = my_data_time_machine.tokenize(train_data[0], token='word')
	vocab = my_data_time_machine.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
	# 在词元化之后，让我们绘制评论词元⻓度的直⽅图。
	my_plt.set_figsize()
	my_plt.plt.xlabel('# tokens per review')
	my_plt.plt.ylabel('count')
	my_plt.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
	my_plt.plt.show()

	# 正如我们所料，评论的⻓度各不相同。
	# 为了每次处理⼀⼩批量这样的评论，我们通过截断和填充将每个评论的⻓度设置为500。
	# 这类似于 9.5节中对机器翻译数据集的预处理步骤。
	num_steps = 500 # 序列⻓度
	train_features = torch.tensor([test_machine_translation_and_dataset.truncate_pad(
		vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
	print("train_features.shape : ", train_features.shape)

	# 现在我们可以创建数据迭代器了。在每次迭代中，都会返回⼀⼩批量样本。
	train_iter = train_framework.load_array((train_features,
						torch.tensor(train_data[1])), 64)
	for X, y in train_iter:
		print('X:', X.shape, ', y:', y.shape)
		break
	print('⼩批量数⽬：', len(train_iter))

# 最后，我们将上述步骤封装到load_data_imdb函数中。
# 它返回训练和测试数据迭代器以及IMDb评论数据集的词表。
#@save
def load_data_imdb(batch_size, num_steps=500):
	"""返回数据迭代器和IMDb评论数据集的词表"""
	data_dir = my_download.download_extract('aclImdb', 'aclImdb')
	train_data = read_imdb(data_dir, True)
	test_data = read_imdb(data_dir, False)
	train_tokens = my_data_time_machine.tokenize(train_data[0], token='word')
	test_tokens = my_data_time_machine.tokenize(test_data[0], token='word')
	vocab = my_data_time_machine.Vocab(train_tokens, min_freq=5)
	train_features = torch.tensor([test_machine_translation_and_dataset.truncate_pad(
		vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
	test_features = torch.tensor([test_machine_translation_and_dataset.truncate_pad(
		vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
	train_iter = train_framework.load_array((train_features, torch.tensor(train_data[1])),
				batch_size)
	test_iter = train_framework.load_array((test_features, torch.tensor(test_data[1])),
				batch_size,
				is_train=False)
	return train_iter, test_iter, vocab

