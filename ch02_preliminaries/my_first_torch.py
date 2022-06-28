import torch
# print\((.*)\)
# print("\1 : ", \1)

x = torch.arange(12) 
print("x : ", x)

print("x.shape : ", x.shape)
print("x.numel() : ", x.numel())

X = x.reshape(3, 4)
print("X : ", X)

zeros234 = torch.zeros((2, 3, 4))
print("zeros234 : ", zeros234)

ones234 = torch.ones((2, 3, 4))
print("ones234 : ", ones234)

randn34 = torch.randn(3, 4)
print("randn34 : ", randn34)

# python语言最常见的括号有三种，其中：
#   小括号( )：代表tuple元组数据类型。
#   中括号[ ]：代表list列表数据类型。
#   大括号{ }：花括号：代表dict字典数据类型，字典是由键对值组组成。
#             冒号':'分开键和值，逗号','隔开组。
test = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("test : ", test)

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print("x + y, x - y, x * y, x / y, x ** y : ", 
	x + y, x - y, x * y, x / y, x ** y) # **运算符是求幂运算

print("torch.exp(x) : ", torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1) : ", 
	torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

print("X == Y : ", X == Y)
print("X.sum() : ", X.sum())

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print("a, b : ", a, b)

print("X[-1], X[1:3] : ", X[-1], X[1:3])
X[1, 2] = 9 
print("X : ", X)
X[0:2, :] = 12
print("X : ", X)

Z = torch.zeros_like(Y)
print("'id(Z):', id(Z) : ", 'id(Z):', id(Z))
Z[:] = X + Y
print("'id(Z):', id(Z) : ", 'id(Z):', id(Z))

before = id(X)
X += Y
id(X) == before

A = X.numpy()
B = torch.tensor(A)
print("type(A), type(B) : ", type(A), type(B))

a = torch.tensor([3.5])
print("a, a.item(), float(a), int(a) : ", a, a.item(), float(a), int(a))




