import torch

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))

mul_then_add = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
print("mul_then_add : ", mul_then_add)
add_then_mul = torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
print("add_then_mul : ", add_then_mul)

