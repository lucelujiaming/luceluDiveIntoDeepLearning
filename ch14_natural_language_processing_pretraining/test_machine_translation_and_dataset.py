import os
import torch

import random
import collections
import re

import my_download
import my_plt
import train_framework
import my_data_time_machine

# 下载⼀个由Tatoeba项⽬的双语句⼦对114 组成的“英－法”数据集
#@save
my_download.DATA_HUB['fra-eng'] = (my_download.DATA_URL +
	 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')
#@save
def read_data_nmt():
	"""载⼊“英语－法语”数据集"""
	data_dir = my_download.download_extract('fra-eng')
	with open(os.path.join(data_dir, 'fra.txt'), 'r',
				encoding='utf-8') as f:
		return f.read()

if __name__ == '__main__':
	raw_text = read_data_nmt()
	print("raw_text[:75] : ", raw_text[:75])

# 下载数据集后，原始⽂本数据需要经过⼏个预处理步骤。
#@save
def preprocess_nmt(text):
	"""预处理“英语－法语”数据集"""
	def no_space(char, prev_char):
		return char in set(',.!?') and prev_char != ' ' 

	# 使⽤空格替换不间断空格
	# 使⽤⼩写字⺟替换⼤写字⺟
	text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
	# 在单词和标点符号之间插⼊空格
	out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
					for i, char in enumerate(text)]
	return ''.join(out)
if __name__ == '__main__':
	text = preprocess_nmt(raw_text)
	print("text[:80] : ", text[:80])

# 下⾯的tokenize_nmt函数对前num_examples个⽂本序列对进⾏词元，
# 其中每个词元要么是⼀个词，要么是⼀个标点符号。
# 此函数返回两个词元列表：source和target：
#     source[i]是源语⾔（这⾥是英语）第i个⽂本序列的词元列表，
#     target[i]是⽬标语⾔（这⾥是法语）第i个⽂本序列的词元列表。
#@save
def tokenize_nmt(text, num_examples=None):
	"""词元化“英语－法语”数据数据集"""
	source, target = [], []
	for i, line in enumerate(text.split('\n')):
		if num_examples and i > num_examples:
			break
		parts = line.split('\t')
		if len(parts) == 2:
			source.append(parts[0].split(' '))
			target.append(parts[1].split(' '))
	return source, target
if __name__ == '__main__':
	source, target = tokenize_nmt(text)
	print("source[:6], target[:6] : ", source[:6], target[:6])

# 让我们绘制每个⽂本序列所包含的词元数量的直⽅图。
# 在这个简单的“英－法”数据集中，⼤多数⽂本序列的词元数量少于20个。
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
	"""绘制列表⻓度对的直⽅图"""
	my_plt.plt.ion()
	my_plt.set_figsize()
	_, _, patches = my_plt.plt.hist(
			[[len(l) for l in xlist], [len(l) for l in ylist]])
	my_plt.plt.xlabel(xlabel)
	my_plt.plt.ylabel(ylabel)
	for patch in patches[1].patches:
		patch.set_hatch('/')
	my_plt.plt.legend(legend)
	my_plt.plt.ioff()
	my_plt.plt.show()
if __name__ == '__main__':
	show_list_len_pair_hist(['source', 'target'], 
		'# tokens per sequence', 'count', source, target);
	# 我们将出现次数少于2次的低频率词元视为相同的未知（“<unk>”）词元。
	# 我们还指定了额外的特定词元，
	# 例如在⼩批量时⽤于将序列填充到相同⻓度的填充词元（“<pad>”），
	# 以及序列的开始词元（“<bos>”）和结束词元（“<eos>”）。
	# 这些特殊词元在⾃然语⾔处理任务中⽐较常⽤。
	src_vocab = my_data_time_machine.Vocab(source, min_freq=2,
			reserved_tokens=['<pad>', '<bos>', '<eos>'])
	print("len(src_vocab) : ", len(src_vocab))

# 下⾯的truncate_pad函数将截断或填充⽂本序列。
#@save
def truncate_pad(line, num_steps, padding_token):
	"""截断或填充⽂本序列"""
	# 如果⽂本序列的词元数⽬多于num_steps时，
	# 我们将截断⽂本序列时，只取其前num_steps 个词元，并且丢弃剩余的词元。
	if len(line) > num_steps:
		return line[:num_steps] # 截断
	# 如果⽂本序列的词元数⽬少于num_steps时，
	# 我们将继续在其末尾添加特定的padding_token，也就是“<pad>”词元，
	# 直到其⻓度达到num_steps；
	return line + [padding_token] * (num_steps - len(line)) # 填充
	# 这样，每个⽂本序列将具有相同的⻓度，以便以相同形状的⼩批量进⾏加载。

if __name__ == '__main__':
	print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

# 定义⼀个函数，可以将⽂本序列转换成⼩批量数据集⽤于训练。
#@save
def build_array_nmt(lines, vocab, num_steps):
	"""将机器翻译的⽂本序列转换成⼩批量"""
	lines = [vocab[l] for l in lines]
	# 我们将特定的“<eos>”词元添加到所有序列的末尾，⽤于表⽰序列的结束。
	# 当模型通过⼀个词元接⼀个词元地⽣成序列进⾏预测时，
	# ⽣成的“<eos>”词元说明完成了序列输出⼯作。
	lines = [l + [vocab['<eos>']] for l in lines]
	array = torch.tensor([truncate_pad(
		l, num_steps, vocab['<pad>']) for l in lines])
	# 此外，我们还记录了每个⽂本序列的⻓度，统计⻓度时排除了填充词元。
	valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
	return array, valid_len

# 定义load_data_nmt函数来返回数据迭代器，以及源语⾔和⽬标语⾔的两种词表。
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
	"""返回翻译数据集的迭代器和词表"""
	text = preprocess_nmt(read_data_nmt())
	source, target = tokenize_nmt(text, num_examples)
	# 源语⾔词表。
	src_vocab = my_data_time_machine.Vocab(source, min_freq=2,
			reserved_tokens=['<pad>', '<bos>', '<eos>'])
	# ⽬标语⾔词表。
	tgt_vocab = my_data_time_machine.Vocab(target, min_freq=2,
			reserved_tokens=['<pad>', '<bos>', '<eos>'])
	# 将源语⾔词元列表转换成⼩批量。
	src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
	# 将⽬标语⾔词元列表转换成⼩批量。
	tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
	# 构建数据。
	data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
	# 调⽤框架中现有的API来读取数据。返回数据迭代器。
	data_iter = train_framework.load_array(data_arrays, batch_size)
	# 返回数据迭代器，以及源语⾔和⽬标语⾔的两种词表。
	return data_iter, src_vocab, tgt_vocab

if __name__ == '__main__':
	# 下⾯我们读出“英语－法语”数据集中的第⼀个⼩批量数据。
	train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
	for X, X_valid_len, Y, Y_valid_len in train_iter:
		print('X:', X.type(torch.int32))
		print('X的有效⻓度:', X_valid_len)
		print('Y:', Y.type(torch.int32))
		print('Y的有效⻓度:', Y_valid_len)
		break
