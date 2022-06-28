import torch
from torch import nn

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


# 如下是卷积层的⼀个简单应⽤：通过找到像素变化的位置，来检测图像中不同颜⾊的边缘。
# 构造⼀个6 × 8像素的⿊⽩图像。
X = torch.eye((8))
print("⼀个6 × 8像素的单位矩阵图像 : \n", X)
# 构造⼀个⾼度为1、宽度为2的卷积核K。
# 输出Y中的1代表从⽩⾊到⿊⾊的边缘，
# -1代表从⿊⾊到⽩⾊的边缘，其他情况的输出为0。
K = torch.tensor([[1.0, -1.0]])
# 我们对参数X（输⼊）和K（卷积核）执⾏互相关运算。
Y = corr2d(X, K)
print("互相关运算 : \n", Y)
# 将输⼊的⼆维图像转置，再进⾏如上的互相关运算。
print("corr2d(X.t(), K) : \n", corr2d(X.t(), K))
print("corr2d(X.t(), K) : \n", corr2d(X.t(), K.t()))
