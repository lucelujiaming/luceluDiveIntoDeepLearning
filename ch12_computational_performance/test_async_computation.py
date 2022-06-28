import os
import subprocess
import numpy
import torch
from torch import nn

import train_framework
import test_torchscript

# 作为热⾝，考虑⼀个简单问题：我们要⽣成⼀个随机矩阵并将其相乘。
# 让我们在NumPy和PyTorch张量中都这样做，看看它们的区别。
# 请注意，PyTorch的tensor是在GPU上定义的。
# GPU计算热⾝
device = train_framework.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)
with test_torchscript.Benchmark('numpy'):
	for _ in range(10):
		a = numpy.random.normal(size=(1000, 1000))
		b = numpy.dot(a, a)
# 请注意，PyTorch的tensor是在GPU上定义的。
with test_torchscript.Benchmark('torch'):
	for _ in range(10):
		a = torch.randn(size=(1000, 1000), device=device)
		b = torch.mm(a, a)


with test_torchscript.Benchmark():
	for _ in range(10):
		a = torch.randn(size=(1000, 1000), device=device)
		b = torch.mm(a, a)
	# raise AssertionError("Torch not compiled with CUDA enabled")
	# torch.cuda.synchronize(device)

# 让我们看另⼀个简单例⼦，以便更好地理解依赖关系图。
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2 
print("z : ", z)

