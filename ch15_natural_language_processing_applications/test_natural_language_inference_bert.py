import json
import multiprocessing
import os
import torch
from torch import nn

import my_download
import train_framework
import my_data_time_machine
import test_bert
import test_natural_language_inference_and_dataset
# 在本节中，我们将下载⼀个预训练好的⼩版本的BERT，然后对其进⾏微调，
# 以便在SNLI数据集上进⾏⾃然语⾔推断。

my_download.DATA_HUB['bert.base'] = (
	my_download.DATA_URL + 'bert.base.torch.zip', 
	'225d66f04cae318b841a13d32af3acc165f253ac')
my_download.DATA_HUB['bert.small'] = (
	my_download.DATA_URL + 'bert.small.torch.zip', 
	'c72329e68a732bef0452e4b96a1c341c8910f81f')
# 两个预训练好的BERT模型都包含⼀个定义词表的“vocab.json”⽂件
# 和⼀个预训练参数的“pretrained.params”⽂件。
# 我们实现了以下load_pretrained_model函数来加载预先训练好的BERT参数。
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
					num_heads, num_layers, dropout, max_len, devices):
	data_dir = my_download.download_extract(pretrained_model)
	# 定义空词表以加载预定义词表
	vocab = my_data_time_machine.Vocab()
	vocab.idx_to_token = json.load(open(os.path.join(data_dir,
						'vocab.json')))
	vocab.token_to_idx = {token: idx for idx, token in enumerate(
						vocab.idx_to_token)}
	bert = test_bert.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
				ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
				num_heads=4, num_layers=2, dropout=0.2,
				max_len=max_len, key_size=256, query_size=256,
				value_size=256, hid_in_features=256,
				mlm_in_features=256, nsp_in_features=256) 
	# 加载预训练BERT参数
	bert.load_state_dict(torch.load(os.path.join(data_dir,
								'pretrained.params')))
	return bert, vocab

if __name__ == '__main__':
	# 为了便于在⼤多数机器上演⽰，我们将在本节中加载和微调经过预训练BERT的⼩版本（“bert.small”）。
	# 在练习中，我们将展⽰如何微调⼤得多的“bert.base”以显著提⾼测试精度。
	devices = train_framework.try_all_gpus()
	bert, vocab = load_pretrained_model(
		'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
		num_layers=2, dropout=0.1, max_len=512, devices=devices)

	# 如果您的计算资源允许，可以微调⼀个更⼤的预训练BERT模型，该模型与原始的BERT基础模型⼀样⼤。
	# 修改load_pretrained_model函数中的参数设置：将“bert.small”替换为“bert.base”，
	# 将num_hiddens=256、ffn_num_hiddens=512、num_heads=4和num_layers=2的值
	# 分别增加到768、3072、12和12。
	# 通过增加微调迭代轮数（可能还会调优其他超参数），你可以获得⾼于0.86的测试精度吗？
	# bert, vocab = load_pretrained_model(
	# 	'bert.base', num_hiddens=768, ffn_num_hiddens=3072, num_heads=12,
	# 	num_layers=12, dropout=0.1, max_len=512, devices=devices)


# 对于SNLI数据集的下游任务⾃然语⾔推断，我们定义了⼀个定制的数据集类SNLIBERTDataset。
class SNLIBERTDataset(torch.utils.data.Dataset):
	def __init__(self, dataset, max_len, vocab=None):
		all_premise_hypothesis_tokens = [[
			p_tokens, h_tokens] for p_tokens, h_tokens in zip( 
				*[my_data_time_machine.tokenize([s.lower() for s in sentences])
					for sentences in dataset[:2]])]
		self.labels = torch.tensor(dataset[2])
		self.vocab = vocab
		self.max_len = max_len
		(self.all_token_ids, self.all_segments,
		self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
		print('read ' + str(len(self.all_token_ids)) + ' examples')
	def _preprocess(self, all_premise_hypothesis_tokens):
		pool = multiprocessing.Pool(4) # 使⽤4个进程
		# pool = multiprocessing.Pool(1)
		out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
		all_token_ids = [
			token_ids for token_ids, segments, valid_len in out]
		all_segments = [segments for token_ids, segments, valid_len in out]
		valid_lens = [valid_len for token_ids, segments, valid_len in out]
		return (torch.tensor(all_token_ids, dtype=torch.long),
			torch.tensor(all_segments, dtype=torch.long),
			torch.tensor(valid_lens))
	def _mp_worker(self, premise_hypothesis_tokens):
		p_tokens, h_tokens = premise_hypothesis_tokens
		self._truncate_pair_of_tokens(p_tokens, h_tokens)
		tokens, segments = test_bert.get_tokens_and_segments(p_tokens, h_tokens)
		token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
					* (self.max_len - len(tokens))
		segments = segments + [0] * (self.max_len - len(segments))
		valid_len = len(tokens)
		return token_ids, segments, valid_len
	def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
		# 为BERT输⼊中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
		while len(p_tokens) + len(h_tokens) > self.max_len - 3:
			if len(p_tokens) > len(h_tokens):
				p_tokens.pop()
			else:
				h_tokens.pop()
	def __getitem__(self, idx):
		return (self.all_token_ids[idx], self.all_segments[idx],
			self.valid_lens[idx]), self.labels[idx]
	def __len__(self):
		return len(self.all_token_ids)

if __name__ == '__main__':
	# 下载完SNLI数据集后，我们通过实例化SNLIBERTDataset类来⽣成训练和测试样本。
	# 这些样本将在⾃然语⾔推断的训练和测试期间进⾏⼩批量读取。
	# 如果出现显存不⾜错误，请减少“batch_size”。在原始的BERT模型中，max_len=512
	batch_size, max_len, num_workers = 512, 128, 0 # train_framework.get_dataloader_workers()
	data_dir = my_download.download_extract('SNLI')
	train_set = SNLIBERTDataset(
		test_natural_language_inference_and_dataset.read_snli(data_dir, True), 
		max_len, vocab)
	test_set = SNLIBERTDataset(
		test_natural_language_inference_and_dataset.read_snli(data_dir, False), 
		max_len, vocab)
	train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
					num_workers=num_workers)
	test_iter = torch.utils.data.DataLoader(test_set, batch_size,
					num_workers=num_workers)

# ⽤于⾃然语⾔推断的微调BERT只需要⼀个额外的多层感知机，
# 该多层感知机由两个全连接层组成（请参⻅下⾯BERTClassifier类中的self.hidden和self.output）。
# 这个多层感知机将特殊的“<cls>”词元的BERT表⽰进⾏了转换，
# 该词元同时编码前提和假设的信息为⾃然语⾔推断的三个输出：蕴涵、⽭盾和中性。
class BERTClassifier(nn.Module):
	def __init__(self, bert):
		super(BERTClassifier, self).__init__()
		self.encoder = bert.encoder
		self.hidden = bert.hidden
		self.output = nn.Linear(256, 3)
	def forward(self, inputs):
		tokens_X, segments_X, valid_lens_x = inputs
		encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
		return self.output(self.hidden(encoded_X[:, 0, :]))

if __name__ == '__main__':
	net = BERTClassifier(bert)
	lr, num_epochs = 1e-4, 5
	trainer = torch.optim.Adam(net.parameters(), lr=lr)
	loss = nn.CrossEntropyLoss(reduction='none')
	train_framework.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
			devices)

