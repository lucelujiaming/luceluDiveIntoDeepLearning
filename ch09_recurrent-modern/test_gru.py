import torch
from torch import nn

import train_framework
import my_data_time_machine

# 读取时间机器数据集：
batch_size, num_steps = 32, 35
train_iter, vocab = my_data_time_machine.load_data_time_machine(batch_size, num_steps)
# 初始化模型参数。
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    # 从标准差为0.01的⾼斯分布中提取权重，并将偏置项设为0，
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    # 超参数num_hiddens定义隐藏单元的数量，
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    # 实例化与更新⻔、重置⻔、候选隐状态和输出层相关的所有权重和偏置。
    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
# 定义隐状态的初始化函数init_gru_state。
def init_gru_state(batch_size, num_hiddens, device):
    # 返回⼀个形状为（批量⼤⼩，隐藏单元个数）的张量，张量的值全部为零。
    return (torch.zeros((batch_size, num_hiddens), device=device), )
# 现在我们准备定义⻔控循环单元模型，
def gru(inputs, state, params):
    # 模型的架构与基本的循环神经⽹络单元是相同的，
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        # 只是权重更新公式更为复杂。
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# 开始训练。训练结束后，我们分别打印输出训练集的困惑度，
# 以及前缀“time traveler”和“traveler”的预测序列上的困惑度。
vocab_size, num_hiddens, device = len(vocab), 256, train_framework.try_gpu()
num_epochs, lr = 500, 1
model = train_framework.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
train_framework.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
predict = lambda prefix: train_framework.predict_ch8(prefix, 50, model, vocab, device)
print(predict('time traveller'))
print(predict('traveller'))

