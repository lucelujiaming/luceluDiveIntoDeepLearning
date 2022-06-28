import torch
from torch import nn

import test_channels

# 下面解释了如何为2 × 2的输⼊张量计算卷积核为2 × 2的转置卷积。
#    输入                              核张量  
#    0 1   → 卷积核为 2 × 2 转置卷积     0 1
#    2 3                               2 3
#                                      输出
#    0 0 X   X 0 1   X X X   X X X    0  0  1
#  = 0 0 X + X 2 3 + 0 2 X + X 0 3 =  0  4  6 
#    X X X   X X X   4 6 X   X 6 9    4 12  9
# 我们可以对输⼊矩阵X和卷积核矩阵K实现基本的转置卷积运算trans_conv。
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
# 转置卷积通过卷积核“⼴播”输⼊元素，从⽽产⽣⼤于输⼊的输出。
# 我们来构建输⼊张量X和卷积核张量K从⽽验证上述实现输出。
X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print("trans_conv(X, K) : ", trans_conv(X, K))

# 当输⼊X和卷积核K都是四维张量时，我们可以使⽤⾼级API获得相同的结果。
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
print("tconv(X) : ", tconv(X))

# 与常规卷积不同，在转置卷积中，填充被应⽤于的输出（常规卷积将填充应⽤于输⼊）。
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, 
    # 当将⾼和宽两侧的填充数指定为1时，转置卷积的输出中将删除第⼀和最后的⾏与列。
    padding=1, 
    bias=False)
tconv.weight.data = K
print("tconv(X) : ", tconv(X))

# 卷积核为2 × 2，步幅为2的转置卷积。
#    输入                          核张量  
#    0 1   → 卷积核为2×2转置卷积      0 1
#    2 3          (步幅2)           2 3
#                                              输出
#    0 0 X X   X X 0 1   X X X X   X X X X    0 0 0 1
#  = 0 0 X X + X X 2 3 + X X X X + X X X X =  0 0 2 3 
#    X X X X   X X X X   0 2 X X   X X 0 3    0 2 0 3
#    X X X X   X X X X   4 6 X X   X X 6 9    4 6 6 9
# 以下代码可以验证步幅为2的转置卷积的输出。
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print("tconv(X) : ", tconv(X))

# 如果我们将X代⼊卷积层f来输出Y = f(X)，并创建⼀个与f具有相同的超参数、
# 但输出通道数量是X中通道数的转置卷积层g，那么g(Y )的形状将与X相同。
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print("tconv(conv(X)).shape : ", tconv(conv(X)).shape)
print("X.shape : ", X.shape)
print("tconv(conv(X)).shape == X.shape : ", tconv(conv(X)).shape == X.shape)

# 在下⾯的⽰例中，我们定义了⼀个3 × 3的输⼊X和2 × 2卷积核K，然后使⽤corr2d函数计算卷积输出Y。
X = torch.arange(9.0).reshape(3, 3) 
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = test_channels.corr2d(X, K)
print("Y : ", Y)

# 接下来，我们将卷积核K重写为包含⼤量0的稀疏权重矩阵W。
def kernel2matrix(K):
    # 权重矩阵的形状是（4，9），其中⾮0元素来⾃卷积核K。
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W 
W = kernel2matrix(K)
print("W : ", W)

# 逐⾏连结输⼊X，获得了⼀个⻓度为9的⽮量。
# 然后，W的矩阵乘法和向量化的X给出了⼀个⻓度为4的向量。
# 重塑它之后，可以获得与上⾯的原始卷积操作所得相同的结果Y：
# 我们刚刚使⽤矩阵乘法实现了卷积。
Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2)
print("Y : ", Y)

# 同样，我们可以使⽤矩阵乘法来实现转置卷积。
Z = trans_conv(Y, K)
Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3)
print("Z : ", Z)














