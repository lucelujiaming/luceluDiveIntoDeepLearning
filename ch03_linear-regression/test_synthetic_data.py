import math
import time
import numpy as np
import torch
import random
import my_plt

def synthetic_data(w, b, num_examples): #@save
	"""⽣成y=Xw+b+噪声"""
	X  = torch.normal(0, 1, (num_examples, len(w)))
	y  = torch.matmul(X, w) + b 
	y += torch.normal(0, 0.01, y.shape)
	return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

my_plt.set_figsize()
my_plt.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
my_plt.plt.show()

# 该函数接收批量⼤⼩、特征矩阵和标签向量作为输⼊，
# ⽣成⼤⼩为batch_size的⼩批量。每个⼩批量包含⼀组特征和标签。
def data_iter(batch_size, features, labels):
	num_examples = len(features)
	indices = list(range(num_examples))
	# 这些样本是随机读取的，没有特定的顺序
	random.shuffle(indices)
	for i in range(0, num_examples, batch_size):
		batch_indices = torch.tensor(
			indices[i: min(i + batch_size, num_examples)])
		yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
	print("X, y : \n", X, '\n', y)
	break

