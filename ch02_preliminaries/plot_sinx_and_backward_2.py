import torch
import matplotlib.pyplot as plt
import numpy as np
x = torch.linspace(0, 3*np.pi, 128)
x.requires_grad_(True)
y = torch.sin(x)  # y = sin(x)

y.sum().backward()

plt.plot(x.detach(), y.detach(), label='y=sin(x)') 
plt.plot(x.detach(), x.grad, label='∂y/∂x=cos(x)')  # dy/dx = cos(x)
plt.legend(loc='upper right')
plt.show()
