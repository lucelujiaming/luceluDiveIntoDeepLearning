import torch
from torch import nn
import my_plt
import my_timer
import test_softmax

# PyTorch不会隐式地调整输⼊的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状
def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.normal_(m.weight, std=0.01)

if __name__ == '__main__':
	batch_size = 256
	train_iter, test_iter = test_softmax.load_data_fashion_mnist(batch_size)

	net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
	net.apply(init_weights);

	loss = nn.CrossEntropyLoss(reduction='none')
	trainer = torch.optim.SGD(net.parameters(), lr=0.1)

	num_epochs = 10
	test_softmax.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

