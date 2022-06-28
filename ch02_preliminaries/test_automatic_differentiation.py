# print\((.*)\)
# print("\1 : ", \1)
# 自动微分(Automatic Differentiation )的目的是为了求函数在某点的导数值。
# 参见：https://blog.csdn.net/aws3217150/article/details/70214422
import numpy as np
import torch
import my_plt

x = torch.arange(4.0) 
print("x : ", x)

x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)
print("Before y.backward() :x.grad : ", x.grad) # 默认值是None

# y = 2 * (0 * 0 + 1 * 1 + 2 * 2 + 3 * 3)
# y = 2 * xTx
# 这里注意的是，y不仅仅存储了计算结果，还存储了运算使用的公式。
# 用于后续计算梯度。
y = 2 * torch.dot(x, x)
print("y : ", y)

# 函数y = 2xTx关于x的梯度应为4x。
y.backward()
print("AFter y.backward() :x.grad : ", x.grad)
print("x.grad == 4 * x : ", x.grad == 4 * x)

print("----------------清除梯度----------------")
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
# 这里把梯度清掉。
x.grad.zero_()
# 重新设置y的值。
y = x.sum()
print("before x.grad : ", x.grad)
y.backward()
print("after  x.grad : ", x.grad)

x.grad.zero_()
# y = x * x。
y = x * x
# 等价于y.backward(torch.ones(len(x)))
print("before x.grad : ", x.grad)
# 调用backward计算反向导数。
# 也就是对于x的每一个元素xi，f'(xi)的值。也被称为梯度值。
y.sum().backward()
# x * x的梯度就是2x。
print("after  x.grad : ", x.grad)

print("----------------分离计算----------------")
# 这里把梯度清掉。
x.grad.zero_()
# y = x * x。
y = x * x 
print("before y.grad : ", y.grad)
# u = y。但是丢弃梯度。变成常数。
u = y.detach()
print("before u.grad : ", u.grad, " and u = ", u)
z = u * x 
print("before z.grad : ", z.grad)
# 梯度
z.sum().backward()
# z.sum() = z.sum(u * x)
# z.sum().grad = u * x = u
# u = {0 ,1 ,4 ,9}
print("after  z.grad : ", z.grad)
print("after  x.grad : ", x.grad)
print("after  x.grad == u : ", x.grad == u)

def f(a):
	b = a * 2
	while b.norm() < 1000: 
		b = b * 2
	if b.sum() > 0: 
		c = b
	else:
		c = 100 * b
	return c

a = torch.randn(size=(), requires_grad=True) 
d = f(a)
d.backward()

print("a.grad == d / a : ", a.grad == d / a)
