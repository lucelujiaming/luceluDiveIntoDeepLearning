import torch
from torch import nn
import train_framework
import my_data_time_machine

# 实现多层循环神经⽹络所需的许多逻辑细节在⾼级API中都是现成的。
# 简单起⻅，我们仅⽰范使⽤此类内置函数的实现⽅式。

batch_size, num_steps = 32, 35
train_iter, vocab = my_data_time_machine.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = train_framework.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = train_framework.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
train_framework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)