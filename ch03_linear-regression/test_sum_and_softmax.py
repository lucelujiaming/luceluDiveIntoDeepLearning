import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import my_plt
import my_timer

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("X.sum(0, keepdim=True), X.sum(1, keepdim=True) : ", 
	X.sum(0, keepdim=True), X.sum(1, keepdim=True))

def softmax(X):
	X_exp = torch.exp(X)
	partition = X_exp.sum(1, keepdim=True)
	return X_exp / partition # 这⾥应⽤了⼴播机制

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print("X_prob, X_prob.sum(1) : ", X_prob, X_prob.sum(1))
