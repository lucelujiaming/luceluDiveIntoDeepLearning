import numpy as np
import torch
from torch import nn

import my_timer

timer = my_timer.Timer()
A = torch.zeros(256, 256) 
B = torch.randn(256, 256) 
C = torch.randn(256, 256)

# 按元素分配只需遍历分别为B和C的所有⾏和列，即可将该值分配给A。 
# 逐元素计算A=BC
timer.start()
for i in range(256):
	for j in range(256):
		A[i, j] = torch.dot(B[i, :], C[:, j])
timeComsumed = timer.stop()
print("timeComsumed : ", timeComsumed)

# 更快的策略是执⾏按列分配。
# 逐列计算A=BC
timer.start()
for j in range(256):
	A[:, j] = torch.mv(B, C[:, j])
timeComsumed = timer.stop()
print("timeComsumed : ", timeComsumed)

# 最有效的⽅法是在⼀个区块中执⾏整个操作。
# ⼀次性计算A=BC
timer.start()
A = torch.mm(B, C)
timeComsumed = timer.stop()
print("timeComsumed : ", timeComsumed)

# 乘法和加法作为单独的操作（在实践中融合）
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, ' 
	f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')

# 我们执⾏相同的矩阵-矩阵乘法，但是这次我们将其⼀次性分为64列的“⼩批量”。
timer.start()
for j in range(0, 256, 64):
	A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timeComsumed = timer.stop()
print("timeComsumed : ", timeComsumed)
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')




