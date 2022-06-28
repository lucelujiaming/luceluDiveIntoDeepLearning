import torch
from torch import nn

import train_framework
import test_similarity_analogy
import test_sentiment_analysis_and_dataset

batch_size = 64
train_iter, test_iter, vocab = test_sentiment_analysis_and_dataset.load_data_imdb(batch_size)

# 在下⾯的BiRNN类中，
class BiRNN(nn.Module):
	def __init__(self, vocab_size, embed_size, num_hiddens,
				num_layers, **kwargs):
		super(BiRNN, self).__init__(**kwargs)
		# 虽然⽂本序列的每个词元经由嵌⼊层（self.embedding）获得其单独的预训练GloVe表⽰，
		self.embedding = nn.Embedding(vocab_size, embed_size)
		# 但是整个序列由双向循环神经⽹络（self.encoder）编码。
		# 更具体地说，双向⻓短期记忆⽹络在初始和最终时间步的隐状态
		#（在最后⼀层）被连结起来作为⽂本序列的表⽰。
		# 将bidirectional设置为True以获取双向循环神经⽹络
		self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
							bidirectional=True)
		# 然后，通过⼀个具有两个输出（“积极”和“消极”）的全连接层（self.decoder），
		# 将此单⼀⽂本表⽰转换为输出类别。
		self.decoder = nn.Linear(4 * num_hiddens, 2)
	def forward(self, inputs):
		# inputs的形状是（批量⼤⼩，时间步数）
		# 因为⻓短期记忆⽹络要求其输⼊的第⼀个维度是时间维，
		# 所以在获得词元表⽰之前，输⼊会被转置。
		# 输出形状为（时间步数，批量⼤⼩，词向量维度）
		embeddings = self.embedding(inputs.T)
		self.encoder.flatten_parameters()
		# 返回上⼀个隐藏层在不同时间步的隐状态，
		# outputs的形状是（时间步数，批量⼤⼩，2*隐藏单元数）
		outputs, _ = self.encoder(embeddings)
		# 连结初始和最终时间步的隐状态，作为全连接层的输⼊，
		# 其形状为（批量⼤⼩，4*隐藏单元数）
		encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
		outs = self.decoder(encoding)
		return outs

# 让我们构造⼀个具有两个隐藏层的双向循环神经⽹络来表⽰单个⽂本以进⾏情感分析。
embed_size, num_hiddens, num_layers = 100, 100, 2
devices = train_framework.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

# 初始化权重。
def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight)
	if type(m) == nn.LSTM:
		for param in m._flat_weights_names:
			if "weight" in param:
				nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights);

# 下⾯，我们为词表中的单词加载预训练的100维（需要与embed_size⼀致）的GloVe嵌⼊。
glove_embedding = test_similarity_analogy.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
# 打印词表中所有词元向量的形状。
print("embeds.shape : ", embeds.shape)
# 我们使⽤这些预训练的词向量来表⽰评论中的词元，并且在训练期间不要更新这些向量。
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False

# 现在我们可以训练双向循环神经⽹络进⾏情感分析。
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")

from pathlib import Path
if __name__ == '__main__':
	if Path("train_ch13_rnn_for_5_times.module").is_file():
		net = torch.load('train_ch13_rnn_for_5_times.module')
	else:
		# 在预训练过程中，我们可以绘制出遮蔽语⾔模型损失和下⼀句预测损失。
		train_framework.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
				devices)
		torch.save(net, 'train_ch13_rnn_for_5_times.module')

# 我们定义以下函数来使⽤训练好的模型net预测⽂本序列的情感。
#@save
def predict_sentiment(net, vocab, sequence):
	"""预测⽂本序列的情感"""
	sequence = torch.tensor(vocab[sequence.split()], device=train_framework.try_gpu())
	label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
	return 'positive' if label == 1 else 'negative'
# 最后，让我们使⽤训练好的模型对两个简单的句⼦进⾏情感预测。
predict_sentiment(net, vocab, 'this movie is so great')

