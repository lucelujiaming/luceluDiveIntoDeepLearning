import os
import random
import torch

import my_download
import train_framework
import my_data_time_machine
import test_bert

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#@save
my_download.DATA_HUB['wikitext-2'] = ( 
	'https://s3.amazonaws.com/research.metamind.io/wikitext/' 
	'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')
#@save
def _read_wiki(data_dir):
	file_name = os.path.join(data_dir, 'wiki.train.tokens')
	with open(file_name, 'r') as f:
		lines = f.readlines()
	# 为了简单起⻅，我们仅使⽤句号作为分隔符来拆分句⼦。
	# ⼤写字⺟转换为⼩写字⺟
	paragraphs = [line.strip().lower().split(' . ')
		for line in lines if len(line.split(' . ')) >= 2]
	random.shuffle(paragraphs)
	return paragraphs

# _get_next_sentence函数⽣成⼆分类任务的训练样本。
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
	# 为了帮助理解两个⽂本序列之间的关系，BERT在预训练中考虑了⼀个⼆元分类任务——下⼀句预测。
	# 在为预训练⽣成句⼦对时，有⼀半的时间它们确实是标签为“真”的连续句⼦；
	if random.random() < 0.5:
		is_next = True
	# 在另⼀半的时间⾥，第⼆个句⼦是从语料库中随机抽取的，标记为“假”。
	else:
		# paragraphs是三重列表的嵌套
		next_sentence = random.choice(random.choice(paragraphs))
		is_next = False
	return sentence, next_sentence, is_next

# 下⾯的函数通过调⽤_get_next_sentence函数从输⼊paragraph⽣成⽤于下⼀句预测的训练样本。
# ⾃变量max_len指定预训练期间的BERT输⼊序列的最⼤⻓度。
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
	nsp_data_from_paragraph = []
	# 这⾥paragraph是句⼦列表，其中每个句⼦都是词元列表。
	for i in range(len(paragraph) - 1):
		tokens_a, tokens_b, is_next = _get_next_sentence(
		paragraph[i], paragraph[i + 1], paragraphs)
		# 考虑1个'<cls>'词元和2个'<sep>'词元
		if len(tokens_a) + len(tokens_b) + 3 > max_len:
			continue
		tokens, segments = test_bert.get_tokens_and_segments(tokens_a, tokens_b)
		nsp_data_from_paragraph.append((tokens, segments, is_next))
	return nsp_data_from_paragraph

# 为了从BERT输⼊序列⽣成遮蔽语⾔模型的训练样本，我们定义了以下_replace_mlm_tokens函数。
# 包含四个参数：
#    tokens                   - 表⽰BERT输⼊序列的词元的列表
#    candidate_pred_positions - 不包括特殊词元的BERT输⼊序列的词元索引的列表
#                              （特殊词元在遮蔽语⾔模型任务中不被预测）
#    num_mlm_preds            - 预测的数量（选择15%要预测的随机词元）。
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
						vocab):
	# 为遮蔽语⾔模型的输⼊创建新的词元副本，其中输⼊可能包含替换的“<mask>”或随机词元
	mlm_input_tokens = [token for token in tokens]
	pred_positions_and_labels = []
	# 打乱后⽤于在遮蔽语⾔模型任务中获取15%的随机词元进⾏预测
	random.shuffle(candidate_pred_positions)
	for mlm_pred_position in candidate_pred_positions:
		if len(pred_positions_and_labels) >= num_mlm_preds:
			break
		masked_token = None
		# 在每个预测位置，
		# 为了避免预训练和微调之间的这种不匹配，如果为预测⽽屏蔽词元
		# （例如，在“this movie is great”中选择掩蔽和预测“great”），则在输⼊中将其替换为：
		# • 80%时间为特殊的“<mask>“词元（例如，“this movie is great”变为“this movie is<mask>”；
		# 80%的时间：将词替换为“<mask>”词元
		if random.random() < 0.8:
			masked_token = '<mask>'
		else:
			# • 10%时间为随机词元（例如，“this movie is great”变为“this movie is drink”）；
			# 10%的时间：保持词不变
			if random.random() < 0.5:
				masked_token = tokens[mlm_pred_position]
			# • 10%时间内为不变的标签词元（例如，“this movie is great”变为“this movie is great”）。
			# 10%的时间：⽤随机词替换该词
			else:
				masked_token = random.choice(vocab.idx_to_token)
		mlm_input_tokens[mlm_pred_position] = masked_token
		pred_positions_and_labels.append(
			(mlm_pred_position, tokens[mlm_pred_position]))
	# 最后，该函数返回可能替换后的输⼊词元、发⽣预测的词元索引和这些预测的标签。
	return mlm_input_tokens, pred_positions_and_labels
# 通过调⽤前述的_replace_mlm_tokens函数，将BERT输⼊序列（tokens）作为输⼊。
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
	candidate_pred_positions = []
	# tokens是⼀个字符串列表
	for i, token in enumerate(tokens):
		# 在遮蔽语⾔模型任务中不会预测特殊词元
		if token in ['<cls>', '<sep>']:
			continue
		candidate_pred_positions.append(i)
	# 遮蔽语⾔模型任务中预测15%的随机词元
	num_mlm_preds = max(1, round(len(tokens) * 0.15))
	mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
					tokens, candidate_pred_positions, num_mlm_preds, vocab)
	pred_positions_and_labels = sorted(pred_positions_and_labels,
					key=lambda x: x[0])
	pred_positions = [v[0] for v in pred_positions_and_labels]
	mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
	# 并返回输⼊词元的索引（在 14.8.5节中描述的可能的词元替换之后）、
	# 发⽣预测的词元索引以及这些预测的标签索引。
	return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


# 我们仍然需要定义辅助函数_pad_bert_inputs来将特殊的“<mask>”词元附加到输⼊。
# 它的参数examples包含来⾃两个预训练任务的辅助函数
# 	_get_nsp_data_from_paragraph和_get_mlm_data_from_tokens的输出。
#@save
def _pad_bert_inputs(examples, max_len, vocab):
	max_num_mlm_preds = round(max_len * 0.15)
	all_token_ids, all_segments, valid_lens, = [], [], []
	all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
	nsp_labels = []
	for (token_ids, pred_positions, mlm_pred_label_ids, segments,
		is_next) in examples:
		all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
			max_len - len(token_ids)), dtype=torch.long))
		all_segments.append(torch.tensor(segments + [0] * (
			max_len - len(segments)), dtype=torch.long))
		# valid_lens不包括'<pad>'的计数
		valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
		all_pred_positions.append(torch.tensor(pred_positions + [0] * (
			max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
		# 填充词元的预测将通过乘以0权重在损失中过滤掉
		all_mlm_weights.append(
			torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
				max_num_mlm_preds - len(pred_positions)),
				dtype=torch.float32))
		all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
			max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
		nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
	return (all_token_ids, all_segments, valid_lens, all_pred_positions,
			all_mlm_weights, all_mlm_labels, nsp_labels)


# 我们定义以下_WikiTextDataset类为⽤于预训练BERT的WikiText-2数据集
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
	# 将⽤于⽣成两个预训练任务的训练样本的辅助函数和⽤于填充输⼊的辅助函数放在⼀起。	
	def __init__(self, paragraphs, max_len):
		# 输⼊paragraphs[i]是代表段落的句⼦字符串列表；
		# ⽽输出paragraphs[i]是代表段落的句⼦列表，其中每个句⼦都是词元列表
		# 为简单起⻅，我们使⽤my_data_time_machine.tokenize函数进⾏词元化。
		# 出现次数少于5次的不频繁词元将被过滤掉。
		paragraphs = [my_data_time_machine.tokenize(
			paragraph, token='word') for paragraph in paragraphs]
		sentences = [sentence for paragraph in paragraphs
					for sentence in paragraph]
		self.vocab = my_data_time_machine.Vocab(sentences, min_freq=5, 
			reserved_tokens=[ '<pad>', '<mask>', '<cls>', '<sep>'])
		# 获取下⼀句⼦预测任务的数据
		examples = []
		for paragraph in paragraphs:
			examples.extend(_get_nsp_data_from_paragraph(
				paragraph, paragraphs, self.vocab, max_len))
		# 获取遮蔽语⾔模型任务的数据
		examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
					+ (segments, is_next))
					for tokens, segments, is_next in examples]
		# 填充输⼊
		(self.all_token_ids, self.all_segments, self.valid_lens,
			self.all_pred_positions, self.all_mlm_weights,
			self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
						examples, max_len, self.vocab)
	# 通过实现__getitem__函数，我们可以任意访问
	# WikiText-2语料库的⼀对句⼦⽣成的预训练样本（遮蔽语⾔模型和下⼀句预测）样本。
	def __getitem__(self, idx):
		return (self.all_token_ids[idx], self.all_segments[idx],
			self.valid_lens[idx], self.all_pred_positions[idx],
			self.all_mlm_weights[idx], self.all_mlm_labels[idx],
			self.nsp_labels[idx])
	def __len__(self):
		return len(self.all_token_ids)

# 通过使⽤_read_wiki函数和_WikiTextDataset类，
# 我们定义了下⾯的load_data_wiki来下载并⽣成WikiText-2数据集，并从中⽣成预训练样本。
#@save
def load_data_wiki(batch_size, max_len):
	"""加载WikiText-2数据集"""
	# num_workers = train_framework.get_dataloader_workers()
	num_workers = 0
	data_dir = my_download.download_extract('wikitext-2', 'wikitext-2')
	paragraphs = _read_wiki(data_dir)
	train_set = _WikiTextDataset(paragraphs, max_len)
	train_iter = torch.utils.data.DataLoader(train_set, batch_size,
					shuffle=True, num_workers=num_workers)
	return train_iter, train_set.vocab

# 将批量⼤⼩设置为512，将BERT输⼊序列的最⼤⻓度设置为64，我们打印出⼩批量的BERT预训练样本的形状。
# 注意，在每个BERT输⼊序列中，为遮蔽语⾔模型任务预测10（64 × 0.15）个位置。
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)
for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
		mlm_Y, nsp_y) in train_iter:
	print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
		pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
		nsp_y.shape)
	break
if __name__ == '__main__':
	print("len(vocab) : ", len(vocab))




