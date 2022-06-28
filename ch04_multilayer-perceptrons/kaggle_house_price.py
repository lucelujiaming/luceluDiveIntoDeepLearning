import hashlib
import os
import tarfile
import zipfile
import requests
import train_framework

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 下载数据集，将数据集缓存在本地⽬录（默认情况下为../data）中，并返回下载⽂件的名称。
def download(name, cache_dir=os.path.join('..', 'data')): #@save
	"""下载⼀个DATA_HUB中的⽂件，返回本地⽂件名"""
	assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
	# 得到下载地址和文件的sha-1值。
	url, sha1_hash = DATA_HUB[name]
	os.makedirs(cache_dir, exist_ok=True)
	fname = os.path.join(cache_dir, url.split('/')[-1])
	# 如果缓存⽬录中已经存在此数据集⽂件，
	if os.path.exists(fname):
		sha1 = hashlib.sha1()
		with open(fname, 'rb') as f:
			while True:
				data = f.read(1048576)
				if not data:
					break
				sha1.update(data)
				# 并且其sha-1与存储在DATA_HUB中的相匹配，
				# 我们将使⽤缓存的⽂件，以避免重复的下载。
				if sha1.hexdigest() == sha1_hash:
					return fname # 命中缓存
	# 如果文件不存在，就下载。
	print(f'正在从{url}下载{fname}...')
	r = requests.get(url, stream=True, verify=True)
	with open(fname, 'wb') as f:
		f.write(r.content)
	return fname

# 下载并解压
def download_extract(name, folder=None): #@save
	"""下载并解压zip/tar⽂件"""
	fname = download(name)
	base_dir = os.path.dirname(fname)
	# 得到扩展名。
	data_dir, ext = os.path.splitext(fname)
	# 如果是zip文件。调用zipfile解压。
	if ext == '.zip':
		fp = zipfile.ZipFile(fname, 'r')
	# 如果是gz文件。调用tarfile解压。
	elif ext in ('.tar', '.gz'):
		fp = tarfile.open(fname, 'r')
	else:
		assert False, '只有zip/tar⽂件可以被解压缩'
	# 执行解压。
	fp.extractall(base_dir)
	return os.path.join(base_dir, folder) if folder else data_dir
# 下载DATA_HUB中的多个文件。
def download_all(): #@save
	"""下载DATA_HUB中的所有⽂件"""
	for name in DATA_HUB:
		download(name)

# 如果你没有安装pandas，请取消下⼀⾏的注释
# !pip install pandas
import numpy as np
import pandas as pd
import torch
from torch import nn

DATA_HUB['kaggle_house_train'] = ( #@save
	DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = ( #@save
	DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
# 使⽤pandas分别加载包含训练数据和测试数据的两个CSV⽂件。
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

print(train_data.shape)
print(test_data.shape)
# 查看前四个和最后两个特征，以及相应标签（房价）。
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# 将第⼀个特征ID从数据集中删除。
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

print(all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 若⽆法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指⽰符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print("all_features.shape : ", all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
		train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
print("train_features.shape", train_features.shape)
loss = nn.MSELoss()
in_features = train_features.shape[1]
# ⾸先，我们训练⼀个带有损失平⽅的线性模型。
def get_network():
	netLinearModel = nn.Sequential(nn.Linear(in_features,1))
	return netLinearModel

# ⽤价格预测的对数来衡量差异。
def log_rmse(net, features, labels):
	# 为了在取对数时进⼀步稳定该值，将⼩于1的值设置为1
	clipped_preds = torch.clamp(net(features), 1, float('inf'))
	# 首先调用loss函数计算features和labels的损失。之后调用torch.sqrt开平方。
	rmse = torch.sqrt(loss(torch.log(clipped_preds),
				torch.log(labels)))
	return rmse.item()
# 与前⾯的部分不同，我们的训练函数将借助Adam优化器。
# Adam优化器的主要吸引⼒在于它对初始学习率不那么敏感。
def train(net, train_features, train_labels, test_features, test_labels,
				num_epochs, learning_rate, weight_decay, batch_size):
	train_ls, test_ls = [], []
	train_iter = train_framework.load_array((train_features, train_labels), batch_size)
	# 这⾥使⽤的是Adam优化算法
	optimizer = torch.optim.Adam(net.parameters(),
						lr = learning_rate,
						weight_decay = weight_decay)
	# 开始循环。
	for epoch in range(num_epochs):
		for X, y in train_iter:
			optimizer.zero_grad()
			# 计算损失。
			l = loss(net(X), y)
			# 更新梯度。
			l.backward()
			optimizer.step()
		# 每循环一次，就调用log_rmse计算一个训练数据的差异结果。
		train_ls.append(log_rmse(net, train_features, train_labels))
		# 如果指定了test_labels，更新test_ls。计算一个验证数据的差异结果。
		if test_labels is not None:
			test_ls.append(log_rmse(net, test_features, test_labels))
	# 返回差异结果。
	return train_ls, test_ls
# 返回第i折的数据。包含下面四个参数：
#    K - 分组总数。
#    i - 选择第i个切⽚作为验证数据，其余部分作为训练数据。
#    X, y - 数据和标签。
def get_k_fold_data(k, i, X, y):
	assert k > 1
	# 获取X的第一个维度。这里的数据其实就是一张二维表格。
	# 因此上，这里得到的是
	fold_size = X.shape[0] // k
	X_train, y_train = None, None
	# 循环K次。
	for j in range(k):
		# 取第j条数据。
		idx = slice(j * fold_size, (j + 1) * fold_size)
		X_part, y_part = X[idx, :], y[idx]
		# 如果j == i，放入验证数据集。
		if j == i:
			X_valid, y_valid = X_part, y_part
		# 否则放入训练数据集。
		elif X_train is None:
			X_train, y_train = X_part, y_part
		else:
			X_train = torch.cat([X_train, X_part], 0)
			y_train = torch.cat([y_train, y_part], 0)
	# 返回验证数据集和训练数据集。
	return X_train, y_train, X_valid, y_valid

# K折交叉验证，包含下面四个参数：
#    K - 分组数。
#    X_train, y_train - 测试数据和测试标签。
#    num_epochs - 训练循环次数。
#    learning_rate - 学习率。
#    weight_decay - 学习率衰减。这份代码中，该值被设置为零。
# 在关于学习率的有效技巧中，有一种被称为学习率衰减（learning rate decay）的方法，
# 即随着学习的进行，使学习率逐渐减小。实际上，一开始“多”学，然后逐渐“少”学的方法，
# 在神经网络的学习中经常被使用。参见《深度学习入门》的“6.1.5　AdaGrad”。
#    batch_size - 批量⼤⼩。
# 训练模型时要对数据集进⾏遍历，每次抽取⼀⼩批量样本，并使⽤它们来更新我们的模型。
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
				batch_size):
	train_l_sum, valid_l_sum = 0, 0
	# 循环K次。
	for i in range(k):
		# 每次选择第i个切⽚作为验证数据，其余部分作为训练数据。
		data = get_k_fold_data(k, i, X_train, y_train)
		# 获得深度学习网络。
		netSequential = get_network()
		# 开始训练。
		# 这里涉及到了Python的函数参数的技巧。
		# 在 Python 的函数中经常能看到输入的参数前面有一个或者两个星号。
		# 这两种用法其实都是用来将任意个数的参数导入到 Python 函数中。
		# 单星号（*）：*agrs 表明将所有参数以元组(tuple)的形式导入。
		# 双星号（**）：**kwargs 表明将参数以字典的形式导入。
		# 因此上这里的*data作为get_k_fold_data的返回值。包括四个参数：
		#    X_train, y_train, X_valid, y_valid
		# 对应train的这四个参数：
		#   train_features, train_labels, test_features, test_labels
		train_ls, valid_ls = train(netSequential, *data, num_epochs, learning_rate,
						weight_decay, batch_size)
		# 累计训练数据的差异结果和验证数据的差异结果。
		train_l_sum += train_ls[-1]
		valid_l_sum += valid_ls[-1]
		# 绘制曲线。
		if i == 0:
			# print("train_ls : ", train_ls)
			# print("valid_ls : ", valid_ls)
			train_framework.my_plt.plot(
				list(range(1, num_epochs + 1)), [train_ls, valid_ls],
				xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
				legend=['train', 'valid'], yscale='log')
		print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f},'
				 f'验证log rmse{float(valid_ls[-1]):f}')
	# 对于计算出来的K轮训练数据的差异结果和验证数据的差异结果求均值。
	return train_l_sum / k, valid_l_sum / k

# 使⽤交叉验证进⾏训练。
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
		weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, ' 
		f'平均验证log rmse: {float(valid_l):f}')

# 使⽤所有数据对其进⾏训练。
def train_and_pred(train_features, test_features, train_labels, test_data,
			num_epochs, lr, weight_decay, batch_size):
	netSequential = get_network()
	train_ls, _ = train(netSequential, train_features, train_labels, None, None,
						num_epochs, lr, weight_decay, batch_size)
	train_framework.my_plt.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
	ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
	print(f'训练log rmse：{float(train_ls[-1]):f}') # 将⽹络应⽤于测试集。
	preds = netSequential(test_features).detach().numpy()
	# 将预测保存在CSV⽂件中可以简化将结果上传到Kaggle的过程。
	# 通过这种⽅式获得的模型可以应⽤于测试集。
	# 将其重新格式化以导出到Kaggle
	test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
	submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
	submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
			num_epochs, lr, weight_decay, batch_size)








