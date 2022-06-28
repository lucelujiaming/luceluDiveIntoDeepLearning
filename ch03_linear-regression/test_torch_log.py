import torch

x = torch.tensor([[1,2], [3,4]])

# 这里的log计算的是自然对数ln()。
log_x  = torch.log(x) 
print(log_x)
# ln(1) = 0
# ln(2) = 0.693147180559945
# ln(3) = 1.09861228866811
# ln(4) = 1.386294361119891
# tensor([[0.0000, 0.6931],
#         [1.0986, 1.3863]])

