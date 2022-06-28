from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

# Sequential使用实例
modelExample = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
print("modelExample : ", modelExample)
# Sequential with OrderedDict使用实例
modelOrderedDict = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
print("modelOrderedDict : ", modelOrderedDict)

# 下面是使用 Module 的模板
# class 网络名字(nn.Module):
#     def __init__(self, 一些定义的参数):
#         super(网络名字, self).__init__()
#         self.layer1 = nn.Linear(num_input, num_hidden)
#         self.layer2 = nn.Sequential(...)
#         ...
# 
#         定义需要用的网络层
# 
#     def forward(self, x): # 定义前向传播
#         x1 = self.layer1(x)
#         x2 = self.layer2(x)
#         x = x1 + x2
#         ...
#         return x

# 为了方便比较，我们先用普通方法搭建一个神经网络。
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net1 = Net(1, 10, 1)
print("net1 : ", net1)

net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
print("net2 : ", net2)

# 总结：
# 使用torch.nn.Module，我们可以根据自己的需求改变传播过程，如RNN等
# 如果你需要快速构建或者不需要过多的过程，直接使用torch.nn.Sequential即可。
