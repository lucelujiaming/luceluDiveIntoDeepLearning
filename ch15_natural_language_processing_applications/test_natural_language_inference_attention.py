import torch
from torch import nn
from torch.nn import functional as F

import my_download
import train_framework
import test_similarity_analogy
import test_machine_translation_and_dataset
import test_natural_language_inference_and_dataset
# 从⾼层次上讲，使⽤注意⼒机制的⾃然语⾔推断⽅法由三个联合训练的步骤组成：
#   对⻬、⽐较和汇总。

# 在下⾯的mlp函数中定义的多层感知机。输出维度f由mlp的num_hiddens参数指定。
def mlp(num_inputs, num_hiddens, flatten):
	net = []
	net.append(nn.Dropout(0.2))
	net.append(nn.Linear(num_inputs, num_hiddens))
	net.append(nn.ReLU())
	if flatten:
		net.append(nn.Flatten(start_dim=1))
	net.append(nn.Dropout(0.2))
	net.append(nn.Linear(num_hiddens, num_hiddens))
	net.append(nn.ReLU())
	if flatten:
		net.append(nn.Flatten(start_dim=1))
	return nn.Sequential(*net)

# 我们定义Attend类来计算假设（beta）与输⼊前提A的软对⻬以及前提（alpha）与输⼊假设B的软对⻬。
class Attend(nn.Module):
	def __init__(self, num_inputs, num_hiddens, **kwargs):
		super(Attend, self).__init__(**kwargs)
		self.f = mlp(num_inputs, num_hiddens, flatten=False)
	def forward(self, A, B):
		# A/B的形状：（批量⼤⼩，序列A/B的词元数，embed_size）
		# f_A/f_B的形状：（批量⼤⼩，序列A/B的词元数，num_hiddens）
		f_A = self.f(A)
		f_B = self.f(B)
		# e的形状：（批量⼤⼩，序列A的词元数，序列B的词元数）
		e = torch.bmm(f_A, f_B.permute(0, 2, 1))
		# beta的形状：（批量⼤⼩，序列A的词元数，embed_size），
		# 意味着序列B被软对⻬到序列A的每个词元(beta的第1个维度)
		beta = torch.bmm(F.softmax(e, dim=-1), B)
		# beta的形状：（批量⼤⼩，序列B的词元数，embed_size），
		# 意味着序列A被软对⻬到序列B的每个词元(alpha的第1个维度)
		alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
		return beta, alpha

# 在⽐较步骤中，我们将来⾃⼀个序列的词元的连结（运算符[·, ·]）和
# 来⾃另⼀序列的对⻬的词元送⼊函数g（⼀个多层感知机）：
class Compare(nn.Module):
	def __init__(self, num_inputs, num_hiddens, **kwargs):
		super(Compare, self).__init__(**kwargs)
		self.g = mlp(num_inputs, num_hiddens, flatten=False)
	def forward(self, A, B, beta, alpha):
		V_A = self.g(torch.cat([A, beta], dim=2))
		V_B = self.g(torch.cat([B, alpha], dim=2))
		return V_A, V_B

# 聚合步骤在以下Aggregate类中定义。
class Aggregate(nn.Module):
	def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
		super(Aggregate, self).__init__(**kwargs)
		self.h = mlp(num_inputs, num_hiddens, flatten=True)
		self.linear = nn.Linear(num_hiddens, num_outputs)
	def forward(self, V_A, V_B):
		# 对两组⽐较向量分别求和
		V_A = V_A.sum(dim=1)
		V_B = V_B.sum(dim=1) # 将两个求和结果的连结送到多层感知机中
		Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
		return Y_hat

# 通过将注意步骤、⽐较步骤和聚合步骤组合在⼀起，我们定义了可分解注意⼒模型来联合训练这三个步骤。
class DecomposableAttention(nn.Module):
	def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
			num_inputs_compare=200, num_inputs_agg=400, **kwargs):
		super(DecomposableAttention, self).__init__(**kwargs)
		self.embedding = nn.Embedding(len(vocab), embed_size)
		self.attend = Attend(num_inputs_attend, num_hiddens)
		self.compare = Compare(num_inputs_compare, num_hiddens)
		# 有3种可能的输出：蕴涵、⽭盾和中性
		self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)
	def forward(self, X):
		premises, hypotheses = X 
		A = self.embedding(premises)
		B = self.embedding(hypotheses)
		beta, alpha = self.attend(A, B)
		V_A, V_B = self.compare(A, B, beta, alpha)
		Y_hat = self.aggregate(V_A, V_B)
		return Y_hat

# 现在，我们将在SNLI数据集上对定义好的可分解注意⼒模型进⾏训练和评估。
# 我们从读取数据集开始。
# 下载并读取SNLI数据集。批量⼤⼩和序列⻓度分别设置为256和50。
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = \
	test_natural_language_inference_and_dataset.load_data_snli(
		batch_size, num_steps)

embed_size, num_hiddens, devices = 100, 200, train_framework.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = test_similarity_analogy.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);

# 我们定义了⼀个split_batch_multi_inputs函数以⼩批量接受多个输⼊，如前提和假设。
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")

from pathlib import Path
if __name__ == '__main__':
	if Path("train_ch13_for_4_times.module").is_file():
		net = torch.load('train_ch13_for_4_times.module')
	else:
		# 在预训练过程中，我们可以绘制出遮蔽语⾔模型损失和下⼀句预测损失。
		train_framework.train_ch13(net, train_iter, test_iter, 
					loss, trainer, num_epochs, devices)
		torch.save(net, 'train_ch13_for_4_times.module')

# 最后，定义预测函数，输出⼀对前提和假设之间的逻辑关系。
#@save
def predict_snli(net, vocab, premise, hypothesis):
	"""预测前提和假设之间的逻辑关系"""
	net.eval()
	premise = torch.tensor(vocab[premise], device=train_framework.try_gpu())
	hypothesis = torch.tensor(vocab[hypothesis], device=train_framework.try_gpu())
	label = torch.argmax(net([premise.reshape((1, -1)),
	hypothesis.reshape((1, -1))]), dim=1)
	return 'entailment' if label == 0 else 'contradiction' if label == 1 \
						else 'neutral'
ret = predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
print(ret)















