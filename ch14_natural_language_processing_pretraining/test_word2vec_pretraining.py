import math
import torch
from torch import nn

import my_timer
import my_plt
import train_framework
import test_word_embedding_dataset

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 我们继续实现跳元语法模型。然后，我们将在PTB数据集上使⽤负采样预训练word2vec。
# 来获得该数据集的数据迭代器和词表。
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = test_word_embedding_dataset.load_data_ptb(
					batch_size, max_window_size, num_noise_words)
# 嵌⼊层：
# 嵌⼊层将词元的索引映射到其特征向量。
# 该层的权重是⼀个矩阵，其⾏数等于字典⼤⼩（input_dim），列数等于每个标记的向量维数（output_dim）。
# 在词嵌⼊模型训练之后，这个权重就是我们所需要的。
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, ' 
	f'dtype={embed.weight.dtype})')

# 当⼩批量词元索引的形状为（2，3）时，嵌⼊层返回具有形状（2，3，4）的向量。
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("embed(x) : ", embed(x))

# 在前向传播中，跳元语法模型的输⼊参数包括：
#    center - 形状为（批量⼤⼩，1）的中⼼词索引
#    contexts_and_negatives - 形状为（批量⼤⼩，max_len）的上下⽂与噪声词索引
#                             其中max_len在 14.3.5节中定义。
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
	# 这两个变量⾸先通过嵌⼊层从词元索引转换成向量，
	v = embed_v(center)
	u = embed_u(contexts_and_negatives)
	# 然后它们的批量矩阵相乘（在 10.2.4节中描述）返回
	# 形状为（批量⼤⼩，1，max_len）的输出。
	pred = torch.bmm(v, u.permute(0, 2, 1))
	# 输出中的每个元素是中⼼词向量和上下⽂或噪声词向量的点积。
	return pred

# 让我们为⼀些样例输⼊打印此skip_gram函数的输出形状。
retSkimGram = skip_gram(torch.ones((2, 1), dtype=torch.long),
			torch.ones((2, 4), dtype=torch.long), embed, embed)
print("skip_gram函数的输出形状 : ", retSkimGram.shape)

# 损失函数：我们将使⽤⼆元交叉熵损失。
class SigmoidBCELoss(nn.Module):
	# 带掩码的⼆元交叉熵损失
	def __init__(self):
		super().__init__()
	def forward(self, inputs, target, mask=None):
		out = nn.functional.binary_cross_entropy_with_logits(
			inputs, target, weight=mask, reduction="none")
		return out.mean(dim=1)
loss = SigmoidBCELoss()

#下⾯计算给定变量的⼆进制交叉熵损失。
pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
print("loss : ", 
	loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))

# 下⾯显⽰了如何使⽤⼆元交叉熵损失中的Sigmoid激活函数（以较低效率的⽅式）计算上述结果。
# 我们可以将这两个输出视为两个规范化的损失，在⾮掩码预测上进⾏平均。
def sigmd(x):
	return -math.log(1 / (1 + math.exp(-x)))
print(f'结果: {(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'结果: {(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')

# 字向量维度embed_size被设置为100。
embed_size = 100
# 我们定义了两个嵌⼊层，将词表中的所有单词分别作为中⼼词和上下⽂词使⽤。
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
								embedding_dim=embed_size),
						nn.Embedding(num_embeddings=len(vocab),
								embedding_dim=embed_size))

# 训练阶段代码实现定义如下。由于填充的存在，损失函数的计算与以前的训练函数略有不同。
def train(net, data_iter, lr, num_epochs, device=train_framework.try_gpu()):
	def init_weights(m):
		if type(m) == nn.Embedding:
			nn.init.xavier_uniform_(m.weight)
	net.apply(init_weights)
	net = net.to(device)
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	animator = train_framework.Animator(xlabel='epoch', ylabel='loss',
		xlim=[1, num_epochs])
	# 规范化的损失之和，规范化的损失数
	metric = train_framework.Accumulator(2)
	for epoch in range(num_epochs):
		timer, num_batches = my_timer.Timer(), len(data_iter)
		for i, batch in enumerate(data_iter):
			optimizer.zero_grad()
			center, context_negative, mask, label = [
			data.to(device) for data in batch]

			pred = skip_gram(center, context_negative, net[0], net[1])
			l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
						/ mask.sum(axis=1) * mask.shape[1])
			l.sum().backward()
			optimizer.step()
			metric.add(l.sum(), l.numel())
			if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
				animator.add(epoch + (i + 1) / num_batches,
					(metric[0] / metric[1],))
	print(f'loss {metric[0] / metric[1]:.3f}, ' f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')

# 现在，我们可以使⽤负采样来训练跳元模型。
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
my_plt.plt.show()

# 在训练word2vec模型之后，我们可以使⽤训练好模型中词向量的余弦相似度
# 来从词表中找到与输⼊单词语义最相似的单词。
def get_similar_tokens(query_token, k, embed):
	W = embed.weight.data
	x = W[vocab[query_token]]
	# 计算余弦相似性。增加1e-9以获得数值稳定性
	cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
				torch.sum(x * x) + 1e-9)
	topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
	for i in topk[1:]: # 删除输⼊词
		print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')
get_similar_tokens('chip', 3, net[0])














