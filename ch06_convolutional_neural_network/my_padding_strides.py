import torch
from torch import nn

# 卷积神经⽹络中卷积核的⾼度和宽度通常为奇数，例如1、3、5或7。
# 对于任何⼆维张量X，当满⾜：
#    1. 卷积核的⼤⼩是奇数；
#    2. 所有边的填充⾏数和列数相同；
#    3. 输出与输⼊具有相同⾼度和宽度。
# 则可以得出：
#    输出Y[i, j]是通过以输⼊X[i, j]为中⼼，与卷积核进⾏互相关计算得到的。
# 为了⽅便起⻅，我们定义了⼀个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输⼊和输出提⾼和缩减相应的维数
def comp_conv2d(conv2d, X):
	# 这⾥的（1，1）表⽰批量⼤⼩和通道数都是1 
	X = X.reshape((1, 1) + X.shape)
	Y = conv2d(X)
	# 省略前两个维度：批量⼤⼩和通道
	return Y.reshape(Y.shape[2:])

# 请注意，这⾥每边都填充了1⾏或1列，因此总共添加了2⾏或2列
# 创建⼀个⾼度和宽度为3的⼆维卷积层，并在所有侧边填充1个像素。
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) 
# 给定⾼度和宽度为8的输⼊，则输出的⾼度和宽度也是8。
X = torch.rand(size=(8, 8))
print("comp_conv2d(conv2d, X).shape : ", comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print("comp_conv2d(conv2d, X).shape : ", comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print("comp_conv2d(conv2d, X).shape : ", comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print("comp_conv2d(conv2d, X).shape : ", comp_conv2d(conv2d, X).shape)


