import torch
from torch import nn

import train_framework
import my_data_time_machine

# 读取时间机器数据集：
batch_size, num_steps = 32, 35
train_iter, vocab = my_data_time_machine.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, train_framework.try_gpu()
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = train_framework.RNNModel(gru_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 1
train_framework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
predict = lambda prefix: train_framework.predict_ch8(prefix, 50, model, vocab, device)
print(predict('time traveller'))
print(predict('traveller'))

