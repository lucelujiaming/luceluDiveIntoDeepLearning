import numpy as np
import torch
import my_plt

def f(x):
	return x ** 3 - 1 / x

x = np.arange(0, 3, 0.1)
my_plt.plot(x, [f(x), 4 * x - 4], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

