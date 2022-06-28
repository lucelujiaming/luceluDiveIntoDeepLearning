# print\((.*)\)
# print("\1 : ", \1)

import numpy as np
import torch
import my_plt

def f(x):
	return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
	return (f(x + h) - f(x)) / h 

h = 0.1
for i in range(5):
	print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}') 
	h *= 0.1


x = np.arange(0, 3, 0.1)
my_plt.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])


