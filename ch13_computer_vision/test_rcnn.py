import torch
import torchvision
X = torch.arange(16.).reshape(1, 1, 4, 4)
print("X : ", X)

# 让我们进⼀步假设输⼊图像的⾼度和宽度都是40像素，且选择性搜索在此图像上⽣成了两个提议区域。
# 每个区域由5个元素表⽰：区域⽬标类别、左上⻆和右下⻆的(x, y)坐标。
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])

# 最后，在2 × 2的兴趣区域汇聚层中，每个兴趣区域被划分为⼦窗⼝⽹格，并进⼀步抽取相同形状2 × 2的特征。
torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)




