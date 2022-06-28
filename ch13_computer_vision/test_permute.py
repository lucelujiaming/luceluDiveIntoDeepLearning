import torch

x = torch.linspace(1, 30, steps=30).view(3,2,5)     # 设置一个三维数组
print(x)
print(x.size())				# 查看数组的维数

# 我们可以把这个permute的过程这样理解；把tensor中的各维度上的值和编号对应。
# 变化一：不改变任何参数。
b = x.permute(0,1,2)            # 不改变维度
print("x.permute(0,1,2) : ", b)
print("x.permute(0,1,2).size() : ", b.size())

# 变化二：1与2交换
b = x.permute(0,2,1)             # 每一块的行与列进行交换，即每一块做转置行为
print(b)
print(b.size())

# 变化三：0与1交换
b = x.permute(1,0,2)            # 交换块和行
print(b)
print(b.size())



