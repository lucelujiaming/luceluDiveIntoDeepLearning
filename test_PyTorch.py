# 方案2：使用清华镜像 源。conda  版本可以看大佬写的，我是直接用了 pip安装
# pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

from __future__ import print_function
import torch
#查看安装版本
print("torch.__version__ : ", torch.__version__)
#查看安装路径
print("torch.__path__ : ", torch.__path__)

x = torch.rand(5, 3)
print(x)

