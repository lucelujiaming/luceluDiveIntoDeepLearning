import torch
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True) 
y = torch.relu(x)
# 绘制relu函数。
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach(), y.detach(), 'x', 'relu(x)') #, figsize=(5, 2.5))
# plt.show()
# 绘制relu函数的梯度函数。
plt.figure(figsize=(5, 2.5))
y.backward(torch.ones_like(x), retain_graph=True)
plt.plot(x.detach(), x.grad, 'x', 'grad of relu') #, figsize=(5, 2.5))
plt.show()
