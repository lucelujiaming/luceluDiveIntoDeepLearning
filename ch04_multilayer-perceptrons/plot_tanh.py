import torch
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True) 
# 绘制tanh函数。
y = torch.tanh(x)
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)') #, figsize=(5, 2.5))
plt.show()

# 绘制tanh函数的梯度函数。
# x.grad.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.figure(figsize=(5, 2.5))
plt.plot(x.detach(), x.grad, 'x', 'grad of sigmoid') #, figsize=(5, 2.5))
plt.show()
