import torch
from torch import nn

import train_framework
import my_timer
import test_bert_dataset
import test_bert

if __name__ == '__main__':
	# ⾸先，我们加载WikiText-2数据集作为⼩批量的预训练样本，⽤于遮蔽语⾔模型和下⼀句预测。
	# 批量⼤⼩是512，BERT输⼊序列的最⼤⻓度是64。注意，在原始BERT模型中，最⼤⻓度是512。
	batch_size, max_len = 512, 64
	train_iter, vocab = test_bert_dataset.load_data_wiki(batch_size, max_len)

	# 原始BERT [Devlin et al., 2018]有两个不同模型尺⼨的版本。
	# 为了便于演⽰，我们定义了⼀个⼩的BERT，使⽤了2层、128个隐藏单元和2个⾃注意头。
	net = test_bert.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
				ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
				num_layers=2, dropout=0.2, key_size=128, query_size=128,
				value_size=128, hid_in_features=128, mlm_in_features=128,
				nsp_in_features=128)
	devices = train_framework.try_all_gpus()
	loss = nn.CrossEntropyLoss()

# 在定义训练代码实现之前，我们定义了⼀个辅助函数_get_batch_loss_bert。
# 给定训练样本，该函数计算遮蔽语⾔模型和下⼀句⼦预测任务的损失。
# 请注意，BERT预训练的最终损失是遮蔽语⾔模型损失和下⼀句预测损失的和。
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
						segments_X, valid_lens_x,
						pred_positions_X, mlm_weights_X,
						mlm_Y, nsp_y):
	# 前向传播
	_, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
								valid_lens_x.reshape(-1),
								pred_positions_X)
	# 计算遮蔽语⾔模型损失
	mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
				 mlm_weights_X.reshape(-1, 1)
	mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8) 
	# 计算下⼀句⼦预测任务的损失
	nsp_l = loss(nsp_Y_hat, nsp_y)
	l = mlm_l + nsp_l
	return mlm_l, nsp_l, l

# 通过调⽤上述两个辅助函数，下⾯的train_bert函数定义了在WikiText-2（train_iter）数据集上
# 预训练BERT（net）的过程。训练BERT可能需要很⻓时间。
# 以下函数的输⼊num_steps指定了训练的迭代步数，⽽不是像train_ch13函数那样指定训练的轮数。
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
	net = nn.DataParallel(net, device_ids=devices).to(devices[0])
	trainer = torch.optim.Adam(net.parameters(), lr=0.01)
	step, timer = 0, my_timer.Timer()
	animator = train_framework.Animator(xlabel='step', ylabel='loss',
				xlim=[1, num_steps], legend=['mlm', 'nsp'])
	# 遮蔽语⾔模型损失的和，下⼀句预测任务损失的和，句⼦对的数量，计数
	metric = train_framework.Accumulator(4)
	num_steps_reached = False
	while step < num_steps and not num_steps_reached:
		for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
			mlm_weights_X, mlm_Y, nsp_y in train_iter:
			tokens_X = tokens_X.to(devices[0])
			segments_X = segments_X.to(devices[0])
			valid_lens_x = valid_lens_x.to(devices[0])
			pred_positions_X = pred_positions_X.to(devices[0])
			mlm_weights_X = mlm_weights_X.to(devices[0])
			mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
			trainer.zero_grad()
			timer.start()
			mlm_l, nsp_l, l = _get_batch_loss_bert(
				net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
				pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
			l.backward()
			trainer.step()
			metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
			timer.stop()
			animator.add(step + 1,
					(metric[0] / metric[3], metric[1] / metric[3]))
			step += 1
			if step == num_steps:
				num_steps_reached = True
				break
	print(f'MLM loss {metric[0] / metric[3]:.3f}, ' 
		  f'NSP loss {metric[1] / metric[3]:.3f}')
	print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on ' 
		  f'{str(devices)}')

from pathlib import Path
if __name__ == '__main__':
	if Path("my_train_bert_for_50_times.module").is_file():
		net = torch.load('my_train_bert_for_50_times.module')
	else:
		# 在预训练过程中，我们可以绘制出遮蔽语⾔模型损失和下⼀句预测损失。
		train_bert(train_iter, net, loss, len(vocab), devices, 50)
		torch.save(net, 'my_train_bert_for_50_times.module')

# 在预训练BERT之后，我们可以⽤它来表⽰单个⽂本、⽂本对或其中的任何词元。
# 下⾯的函数返回tokens_a和tokens_b中所有词元的BERT（net）表⽰。
def get_bert_encoding(net, tokens_a, tokens_b=None):
	tokens, segments = test_bert.get_tokens_and_segments(tokens_a, tokens_b)
	token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
	segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
	valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
	encoded_X, _, _ = net(token_ids, segments, valid_len)
	return encoded_X

if __name__ == '__main__':
	# 考虑“a crane is flying”这句话。
	tokens_a = ['a', 'crane', 'is', 'flying']
	# 插⼊特殊标记“<cls>”（⽤于分类）和“<sep>”（⽤于分隔）后，BERT输⼊序列的⻓度为6。
	# 因为零是“<cls>”词元，encoded_text[:,0, :]是整个输⼊语句的BERT表⽰。
	encoded_text = get_bert_encoding(net, tokens_a)
	# 词元：'<cls>','a','crane','is','flying','<sep>'
	encoded_text_cls = encoded_text[:, 0, :]
	encoded_text_crane = encoded_text[:, 2, :]
	print("encoded_text.shape : ", encoded_text.shape)
	print("encoded_text_cls.shape : ", encoded_text_cls.shape)
	# 为了评估⼀词多义词元“crane”，我们还打印出了该词元的BERT表⽰的前三个元素。
	print("encoded_text_crane[0][:3] : ", encoded_text_crane[0][:3])

	# 现在考虑⼀个句⼦“a crane driver came”和“he just left”。
	tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
	# 类似地，encoded_pair[:, 0, :]是来⾃预训练BERT的整个句⼦对的编码结果。
	encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
	# 词元：'<cls>','a','crane','driver','came','<sep>','he','just',
	# 'left','<sep>'
	encoded_pair_cls = encoded_pair[:, 0, :]
	encoded_pair_crane = encoded_pair[:, 2, :]
	print("encoded_text.shape : ", encoded_pair.shape)
	print("encoded_text_cls.shape : ", encoded_pair_cls.shape)
	# 注意，多义词元“crane”的前三个元素与上下⽂不同时的元素不同。这⽀持了BERT表⽰是上下⽂敏感的。
	print("encoded_text_crane[0][:3] : ", encoded_pair_crane[0][:3])






