import torch
# print\((.*)\)
# print("\1 : ", \1)

x = torch.tensor(3.0) 
y = torch.tensor(2.0) 
print("x + y, x * y, x / y, x**y : ", x + y, x * y, x / y, x**y)

x = torch.arange(4) 
print("x : ", x)
print("x[3] : ", x[3])

print("x.shape : ", x.shape)

A = torch.arange(20).reshape(5, 4) 
print("A : ", A)
print("A.T : ", A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print("B : ", B)
print("B == B.T : ", B == B.T)

X = torch.arange(24).reshape(2, 3, 4) 
print("X : ", X)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4) 
B = A.clone() # 通过分配新内存，将A的⼀个副本分配给B
print("A, A + B : ", A, A + B)
print("A * B : ", A * B)

a = 2 
X = torch.arange(24).reshape(2, 3, 4) 
print("a + X, (a * X).shape : ", a + X, (a * X).shape)

x = torch.arange(4, dtype=torch.float32)
print("x, x.sum() : ", x, x.sum())

print("A.shape, A.sum() : ", A.shape, A.sum())

A_sum_axis0 = A.sum(axis=0)
print("A_sum_axis0, A_sum_axis0.shape : ", A_sum_axis0, A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print("A_sum_axis1, A_sum_axis1.shape : ", A_sum_axis1, A_sum_axis1.shape)

print("A.sum(axis=[0, 1]) : ", A.sum(axis=[0, 1])) # SameasA.sum()

print("A.mean(), A.sum() / A.numel() : ", A.mean(), A.sum() / A.numel())

sum_A = A.sum(axis=1, keepdims=True)
print("sum_A : ", sum_A)
print("A / sum_A : ", A / sum_A)

print("A.cumsum(axis=0) : ", A.cumsum(axis=0))

y = torch.ones(4, dtype = torch.float32)
print("x, y, torch.dot(x, y) : ", x, y, torch.dot(x, y))
print("torch.sum(x * y) : ", torch.sum(x * y))

print("A.shape, x.shape, torch.mv(A, x) : ", A.shape, x.shape, torch.mv(A, x))

B = torch.ones(4, 3)
print("orch.mm(A, B) : ", torch.mm(A, B))

u = torch.tensor([3.0, -4.0])
print("torch.norm(u) : ", torch.norm(u))
print("torch.abs(u).sum() : ", torch.abs(u).sum())

print("torch.norm(torch.ones((4, 9))) : ", torch.norm(torch.ones((4, 9))))
