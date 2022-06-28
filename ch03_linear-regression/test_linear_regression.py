import math
import time
import numpy as np
import torch
import my_plt

n = 10000
a = torch.ones(n)
b = torch.ones(n)

class Timer: #@save
	"""记录多次运⾏时间"""
	def __init__(self):
		self.times = []
		self.start()
	def start(self):
		"""启动计时器"""
		self.tik = time.time()
	def stop(self):
		"""停⽌计时器并将时间记录在列表中"""
		self.times.append(time.time() - self.tik)
		return self.times[-1]
	def avg(self):
		"""返回平均时间"""
		return sum(self.times) / len(self.times)
	def sum(self):
		"""返回时间总和"""
		return sum(self.times)
	def cumsum(self):
		"""返回累计时间"""
		return np.array(self.times).cumsum().tolist()

c = torch.zeros(n)
timer = Timer()
for i in range(n):
	c[i] = a[i] + b[i]
print(f'Using time : {timer.stop():.5f} sec')

timer.start()
d = a + b 
print(f'Using time : {timer.stop():.5f} sec')




