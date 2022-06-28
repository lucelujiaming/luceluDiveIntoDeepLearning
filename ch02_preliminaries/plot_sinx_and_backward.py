import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import torch

f, ax = plt.subplots(1)

x = np.linspace(-3 * np.pi, 3 * np.pi, 100)
x1= torch.tensor(x, requires_grad=True)
y1= torch.sin(x1)
y1.sum().backward()

ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, x1.grad,   label="gradient of sin(x)")
ax.legend(loc="upper center", shadow=True)

ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

plt.show()