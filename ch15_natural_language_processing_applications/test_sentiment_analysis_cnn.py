import torch
from torch import nn

import train_framework
import test_similarity_analogy
import test_sentiment_analysis_and_dataset

batch_size = 64
train_iter, test_iter, vocab = test_sentiment_analysis_and_dataset.load_data_imdb(batch_size)

# 在介绍该模型之前，让我们先看看⼀维卷积是如何⼯作的。
# 请记住，这只是基于互相关运算的⼆维卷积的特例。
#      ______输入_____    _核__   ______输出______
#      | | | | | | | |   | | |   | | | |  |  |  |
#      |0|1|2|3|4|5|6| * |2|3| = |2|5|8|11|14|17|
#      |_|_|_|_|_|_|_|   |_|_|   |_|_|_|__|__|__|
# 
# 我们在下⾯的corr1d函数中实现了⼀维互相关。给定输⼊张量X和核张量K，它返回输出张量Y。
def corr1d(X, K):
	w = K.shape[0] 
	Y = torch.zeros((X.shape[0] - w + 1))
	for i in range(Y.shape[0]):
		Y[i] = (X[i: i + w] * K).sum()
	return Y

# 我们可以从上图构造输⼊张量X和核张量K来验证上述⼀维互相关实现的输出。
X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
corr1dRet = corr1d(X, K)
print("corr1d(X, K) : ", corr1dRet)

# 下面演⽰具有3个输⼊通道的⼀维互相关操作。
#      _____输入______   __核___   __________乘积___________
#      | | | | | | | |   |  |  |   |   |   |   |   |   |   |
#      |2|3|4|5|6|7|8|   |-1|-3|   |-11|-15|-19|-23|-27|-31|
#      |_|_|_|_|_|_|_|   |__|__|   |___|___|___|___|___|___|   _______累加输出__________
#      | | | | | | | |   |  |  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
#      |1|2|3|4|5|6|7| * | 3| 4| = | 11| 18| 25| 32| 39| 46| = |  2|  8| 14| 20| 26| 32|
#      |_|_|_|_|_|_|_|   |__|__|   |___|___|___|___|___|___|   |___|___|___|___|___|___|
#      | | | | | | | |   |  |  |   |   |   |   |   |   |   |
#      |0|1|2|3|4|5|6|   | 1| 2|   |  2|  5|  8| 11| 14| 17|
#      |_|_|_|_|_|_|_|   |__|__|   |___|___|_ _|___|___|___|

# 我们可以实现多个输⼊通道的⼀维互相关运算，并在 图15.3.3中验证结果。
def corr1d_multi_in(X, K):
	# ⾸先，遍历'X'和'K'的第0维（通道维）。然后，把它们加在⼀起
	return sum(corr1d(x, k) for x, k in zip(X, K))
X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
[1, 2, 3, 4, 5, 6, 7],
[2, 3, 4, 5, 6, 7, 8]])
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
corr1dRet = corr1d_multi_in(X, K)
print("corr1d_multi_in(X, K) : ", corr1dRet)

# 我们在下⾯的类中实现textCNN模型。
# 与双向循环神经⽹络模型相⽐，除了⽤卷积层代替循环神经⽹络层外，
# 我们还使⽤了两个嵌⼊层：⼀个是可训练权重，另⼀个是固定权重。
class TextCNN(nn.Module):
	# 使⽤⼀维卷积和最⼤时间汇聚，textCNN模型将单个预训练的词元表⽰作为输⼊，
	# 然后获得并转换⽤于下游应⽤的序列表⽰。

	# 对于具有由d维向量表⽰的n个词元的单个⽂本序列，
		# 输⼊张量的宽度、⾼度和通道数分别为n、1和d。textCNN模型将输⼊转换为输出，如下所⽰：
	def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
					**kwargs):
		super(TextCNN, self).__init__(**kwargs)
		self.embedding = nn.Embedding(vocab_size, embed_size)
		# 这个嵌⼊层不需要训练
		self.constant_embedding = nn.Embedding(vocab_size, embed_size)
		# 3. 使⽤全连接层将连结后的向量转换为输出类别。Dropout可以⽤来减少过拟合。
		self.dropout = nn.Dropout(0.5)
		self.decoder = nn.Linear(sum(num_channels), 2) 
		# 2. 在所有输出通道上执⾏最⼤时间汇聚层，然后将所有标量汇聚输出连结为向量。
		# 最⼤时间汇聚层没有参数，因此可以共享此实例
		self.pool = nn.AdaptiveAvgPool1d(1)
		self.relu = nn.ReLU()
		# 1. 定义多个⼀维卷积核，并分别对输⼊执⾏卷积运算。
		#    具有不同宽度的卷积核可以捕获不同数⽬的相邻词元之间的局部特征。
		# 创建多个⼀维卷积层
		self.convs = nn.ModuleList()
		for c, k in zip(num_channels, kernel_sizes):
			self.convs.append(nn.Conv1d(2 * embed_size, c, k))
	def forward(self, inputs):
		# 沿着向量维度将两个嵌⼊层连结起来，
		# 每个嵌⼊层的输出形状都是（批量⼤⼩，词元数量，词元向量维度）连结起来
		embeddings = torch.cat((
			self.embedding(inputs), self.constant_embedding(inputs)), dim=2) 
		# 根据⼀维卷积层的输⼊格式，重新排列张量，以便通道作为第2维
		embeddings = embeddings.permute(0, 2, 1) 
		# 每个⼀维卷积层在最⼤时间汇聚层合并后，获得的张量形状是（批量⼤⼩，通道数，1） 
		# 删除最后⼀个维度并沿通道维度连结

		# 对于具有由d维向量表⽰的n个词元的单个⽂本序列，
		# 输⼊张量的宽度、⾼度和通道数分别为n、1和d。textCNN模型将输⼊转换为输出，如下所⽰：
		# 1. 定义多个⼀维卷积核，并分别对输⼊执⾏卷积运算。
		#    具有不同宽度的卷积核可以捕获不同数⽬的相邻词元之间的局部特征。
		# 2. 在所有输出通道上执⾏最⼤时间汇聚层，然后将所有标量汇聚输出连结为向量。
		encoding = torch.cat([
			torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
			for conv in self.convs], dim=1)		
		# 3. 使⽤全连接层将连结后的向量转换为输出类别。Dropout可以⽤来减少过拟合。
		outputs = self.decoder(self.dropout(encoding))
		return outputs
# 让我们创建⼀个textCNN实例。它有3个卷积层，卷积核宽度分别为3、4和5，均有100个输出通道。
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = train_framework.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

# 初始化权重。
def init_weights(m):
	if type(m) in (nn.Linear, nn.Conv1d):
		nn.init.xavier_uniform_(m.weight)
net.apply(init_weights);

# 我们加载预训练的100维GloVe嵌⼊作为初始化的词元表⽰。
# 这些词元表⽰（嵌⼊权重）在embedding中将被训练，在constant_embedding中将被固定。
glove_embedding = test_similarity_analogy.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False

# 现在我们可以训练textCNN模型进⾏情感分析。
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")

from pathlib import Path
if __name__ == '__main__':
	if Path("train_ch13_cnn_for_5_times.module").is_file():
		net = torch.load('train_ch13_cnn_for_5_times.module')
	else:
		# 在预训练过程中，我们可以绘制出遮蔽语⾔模型损失和下⼀句预测损失。
		train_framework.train_ch13(net, train_iter, test_iter, loss, 
			trainer, num_epochs, devices)
		torch.save(net, 'train_ch13_cnn_for_5_times.module')

# 下⾯，我们使⽤训练好的模型来预测两个简单句⼦的情感。
# 我们定义以下函数来使⽤训练好的模型net预测⽂本序列的情感。
#@save
def predict_sentiment(net, vocab, sequence):
	"""预测⽂本序列的情感"""
	sequence = torch.tensor(vocab[sequence.split()], device=train_framework.try_gpu())
	label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
	return 'positive' if label == 1 else 'negative'
predict_sentiment(net, vocab, 'this movie is so great')
predict_sentiment(net, vocab, 'this movie is so bad')







