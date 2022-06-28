import torch
from torch import nn
import train_framework
import my_data_time_machine

# ⾸先加载时光机器数据集。
batch_size, num_steps = 32, 35
train_iter, vocab = my_data_time_machine.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, train_framework.try_gpu()
num_epochs, lr = 500, 1
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = train_framework.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
train_framework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)