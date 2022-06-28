import torch
from torch import nn

import train_framework
import my_data_time_machine

# 该模型预测未来词元的能⼒却可能存在严重缺陷。
# 我们⽤下⾯的⽰例代码引以为戒，以防在错误的环境中使⽤它们。
# 加载数据
batch_size, num_steps, device = 32, 35, train_framework.try_gpu()
train_iter, vocab = my_data_time_machine.load_data_time_machine(batch_size, num_steps)
# 通过设置“bidirective=True”来定义双向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = train_framework.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# 训练模型
num_epochs, lr = 500, 1
train_framework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)