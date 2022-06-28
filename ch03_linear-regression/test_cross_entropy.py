import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import my_plt
import my_timer

# 数据样本y_hat其中包含2个样本在3个类别的预测概率，以及它们对应的标签y。
# 也就是：
#    样本0在类别0，1，2中的概率为[0.1, 0.3, 0.6]
#    样本1在类别0，1，2中的概率为[0.3, 0.2, 0.5]
# 该数据拥有归一性。
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# 而y是真实值。也就是：
#    样本0属于类别0
#    样本1属于类别2
y = torch.tensor([0, 2])
# 下面的代码使⽤y作为y_hat中概率的索引，
# 我们选择第⼀个样本中第⼀个类的概率和第⼆个样本中第三个类的概率。
# 也就是第一个参数表明下标。也就是获取y_hat的第0个和1个元素，也就是：
# [0.1, 0.3, 0.6], [0.3, 0.2, 0.5]。
# 而第二参数则给出了y_hat的第0个和1个元素的内部元素下标[0, 2]。
# 也就是[0.1, 0.3, 0.6]的第0个元素0.1。
# [0.3, 0.2, 0.5]的第2个元素0.5。
# 结果就是[0.1, 0.5]。
print("y_hat[[0, 1], y] : ", y_hat[[0, 1], y])

# 有了上面的基础，我们来实现交叉熵损失函数。
def cross_entropy(y_hat, y):
	# 这里的log计算的是自然对数ln()。
	return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y): #@save
	"""计算预测正确的数量"""
	# 假定第⼆个维度存储每个类的预测分数。
	if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
		# 使⽤argmax获得每⾏中最⼤元素的索引来获得预测类别。
		y_hat = y_hat.argmax(axis=1)
	# 将预测类别与真实y元素进⾏⽐较。
	# 结果是⼀个包含0（错）和1（对）的张量。
	cmp = y_hat.type(y.dtype) == y
	# 最后，我们求和会得到正确预测的数量。
	return float(cmp.type(y.dtype).sum())
	
print("accuracy(y_hat, y) / len(y) : ", accuracy(y_hat, y) / len(y))


