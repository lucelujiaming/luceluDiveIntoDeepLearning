import torch

# 该函数接受输⼊张量X和卷积核张量K，并返回输出张量Y。
def corr2d(X, K):  #@save
    """计算二维互相关运算。"""
    # 得到卷积核张量K的大小。
    h, w = K.shape
    # 根据输入张量X和卷积核张量K的大小，创建输出张量Y，初始化为零。
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 进行卷积运算。
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# 多输⼊通道互相关运算，简⽽⾔之，我们所做的就是对每个通道执⾏互相关操作，然后将结果相加。
def corr2d_multi_in(X, K):
	# 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在⼀起
	return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
				  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

if __name__ == '__main__':
	print("corr2d_multi_in(X, K) : ", corr2d_multi_in(X, K))

# 实现⼀个计算多个通道的输出的互相关函数。
def corr2d_multi_in_out(X, K):
	# 迭代“K”的第0个维度，每次都对输⼊“X”执⾏互相关运算。
	# 最后将所有结果都叠加在⼀起
	return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

if __name__ == '__main__':
	# 通过将核张量K与K+1（K中每个元素加1）和K+2连接起来，构造了⼀个具有3个输出通道的卷积核。
	K = torch.stack((K, K + 1, K + 2), 0) 
	print("K.shape : ", K.shape)
	# 对输⼊张量X与卷积核张量K执⾏互相关运算。
	# 现在的输出包含3个通道，第⼀个通道的结果与先前输⼊张量X和多输⼊单输出通道的结果⼀致。
	print("corr2d_multi_in_out(X, K) : ", corr2d_multi_in_out(X, K))

# 使⽤全连接层实现1 × 1卷积。请注意，我们需要对输⼊和输出的数据形状进⾏调整。
def corr2d_multi_in_out_1x1(X, K):
	# 获得X的大小。X为三维数据。
	c_i, h, w = X.shape
	# 获得K的高度。K为三维数据。
	c_o = K.shape[0] 
	# 把X摊平成二维矩阵。
	X = X.reshape((c_i, h * w))
	# 把K摊平成二维矩阵。
	K = K.reshape((c_o, c_i))
	# 全连接层中的矩阵乘法
	Y = torch.matmul(K, X)
	# 计算出来的结果，恢复成三维数据，其中高度等于K的高度。
	return Y.reshape((c_o, h, w))
	
if __name__ == '__main__':
	# 当执⾏1 × 1卷积运算时，上述函数相当于先前实现的互相关函数corr2d_multi_in_out。
	# ⽤⼀些样本数据来验证这⼀点。
	X = torch.normal(0, 1, (3, 3, 3))
	K = torch.normal(0, 1, (2, 3, 1, 1))
	Y1 = corr2d_multi_in_out_1x1(X, K)
	Y2 = corr2d_multi_in_out(X, K)
	# assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
	print("float(torch.abs(Y1 - Y2).sum()) < 1e-6 : ", 
		float(torch.abs(Y1 - Y2).sum()) < 1e-6)

