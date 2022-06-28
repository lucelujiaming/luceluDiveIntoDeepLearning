# 设计⼀个⽬标检测模型：单发多框检测（SSD）
# 此模型主要由基础⽹络组成，其后是⼏个多尺度特征块。
# 基本⽹络⽤于从输⼊图像中提取特征，因此它可以使⽤深度卷积神经⽹络。现在也常⽤ResNet替代。

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

import test_anchor
import train_framework
import test_object_detection_dataset
import my_timer
import my_plt

# 类别预测层：
#   设⽬标类别的数量为q。这样⼀来，锚框有q + 1个类别，其中0类是背景。
#   在某个尺度下，设特征图的⾼和宽分别为h和w。如果以其中每个单元为中⼼⽣成a个锚框。
#   我们需要对hwa个锚框进⾏分类。因此上，单发多框检测采⽤⽤卷积层的通道来降低模型复杂度。

# 类别预测层使⽤⼀个保持输⼊⾼和宽的卷积层。
# 这样⼀来，输出和输⼊在特征图宽和⾼上的空间坐标⼀⼀对应。
# 因此输出通道数为a(q + 1)，其中索引为i(q + 1) + j（0 ≤ j ≤ q）的通道
# 代表了索引为i的锚框有关类别索引为j的预测。

# 定义了这样⼀个类别预测层：
#    参数num_anchors指定了a。
#    参数num_classes指定了q。
# 该图层使⽤填充为1的3 × 3的卷积层。此卷积层的输⼊和输出的宽度和⾼度保持不变。
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
        kernel_size=3, padding=1)
    
# 边界框预测层的设计与类别预测层的设计类似。
# 唯⼀不同的是，这⾥需要为每个锚框预测4个偏移量，⽽不是q + 1个类别。
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# 为同⼀个⼩批量构建两个不同⽐例（Y1和Y2）的特征图，
def forward(x, block):
    return block(x)
Y1 = forward(
    # 其中Y2的⾼度和宽度是Y1的⼀半。
    torch.zeros((2, 8, 20, 20)), 
    # 以类别预测为例，假设Y1和Y2的每个单元分别⽣成了5个和3个锚框。
    # 进⼀步假设⽬标类别的数量为10，
    cls_predictor(8, 5, 10))
Y2 = forward(
    torch.zeros((2, 16, 10, 10)), 
    cls_predictor(16, 3, 10))

if __name__ == '__main__':
    # 其中任⼀输出的形状是（批量⼤⼩，通道数，⾼度，宽度）。
    # 对于特征图Y1，类别预测输出中的通道数为5 × (10 + 1) = 55
    # Y1.shape :  torch.Size([2, 55, 20, 20]) 
    print("Y1.shape : ", Y1.shape)
    # 对于特征图Y2，类别预测输出中的通道数为3 × (10 + 1) = 33，
    # Y2.shape :  torch.Size([2, 33, 10, 10])
    print("Y2.shape : ", Y2.shape)

# 我们⾸先将通道维移到最后⼀维。因为不同尺度下批量⼤⼩仍保持不变，
def flatten_pred(pred):
    # 之前的形状是（批量⼤⼩，通道数，⾼度，宽度）。
    # permute后为（批量⼤⼩，⾼度，宽度，通道数）。
    # 之后我们调用torch.flatten(XXX, start_dim=1)进行展平。
    # 这里的(start_dim=1)表明保留第一个维度。维度变成（批量⼤⼩，N）。
    # 另外，如果(start_dim=1)就会变成单个数字的list。
    # 因为维度变成（批量⼤⼩，N），不同大小的图片就可以连接起来。
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)
# 我们可以将预测结果转成⼆维的（批量⼤⼩，⾼×宽×通道数）的格式，以⽅便之后在维度1上的连结。
def concat_preds(preds):
    # 这里一次传入两个不同⽐例（Y1和Y2）的特征图。
    # 函数针对每一个特征图调用flatten_pred方法，
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

if __name__ == '__main__':
    # 我们在同⼀个⼩批量的两个不同尺度上连接这两个预测输出。
    print("concat_preds([Y1, Y2]).shape", concat_preds([Y1, Y2]).shape)

# 为了在多个尺度下检测⽬标，该模块将输⼊特征图的⾼度和宽度减半。
# 该块应⽤了在 subsec_vgg-blocks中的VGG模块设计。
def down_sample_blk(in_channels, out_channels):
    blk = []
    # 每个⾼和宽减半块由两个：
    for _ in range(2):
        # 填充为1的3×3的卷积层、
        # （填充为1的3×3卷积层不改变特征图的形状）
        blk.append(nn.Conv2d(in_channels, out_channels,
            kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    # 以及一个步幅为2的2×2最⼤汇聚层组成。
    # （2 × 2的最⼤汇聚层将输⼊特征图的⾼度和宽度减少了⼀半）
    # 对于此⾼和宽减半块的输⼊和输出特征图，因为1 × 2 + (3 − 1) + (3 − 1) = 6，
    # 所以输出中的每个单元在输⼊上都有⼀个6 × 6的感受野。
    # 因此，⾼和宽减半块会扩⼤每个单元在其输出特征图中的感受野。
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

if __name__ == '__main__':
    # 在以下⽰例中，我们构建的⾼和宽减半块会更改输⼊通道的数量，并将输⼊特征图的⾼度和宽度减半。
    forwardRet = forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10))
    print("down_sample_blk(forwardRet) : ", forwardRet.shape)

# 基本⽹络块⽤于从输⼊图像中抽取特征。
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)
if __name__ == '__main__':
    # 为了计算简洁，我们构造了⼀个⼩的基础⽹络，该⽹络串联3个⾼和宽减半块，并逐步将通道数翻倍。
    # 给定输⼊图像的形状为256×256，此基本⽹络块输出的特征图形状为32×32（256/23 = 32）。
    forwardRet = forward(torch.zeros((2, 3, 256, 256)), base_net())
    print("base_net(forwardRet) : ", forwardRet.shape)

# 完整的单发多框检测模型由五个模块组成。
# 每个块⽣成的特征图既⽤于⽣成锚框，⼜⽤于预测这些锚框的类别和偏移量。
# 最后⼀个模块使⽤全局最⼤池将⾼度和宽度都降到1。
def get_blk(i):
    # 在这五个模块中，第⼀个是基本⽹络块，
    if i == 0:
        blk = base_net()
    # 第⼆个到第四个是⾼和宽减半块，
    elif i == 1:
        blk = down_sample_blk(64, 128)
    # 最后⼀个模块使⽤全局最⼤池将⾼度和宽度都降到1。
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    # 第⼆个到第四个是⾼和宽减半块，
    else:
        blk = down_sample_blk(128, 128)
    return blk

# 现在我们为每个块定义前向传播。
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    # 与图像分类任务不同，此处的输出包括：
    #      CNN特征图Y；
    Y = blk(X)
    # 在每个多尺度特征块上，我们通过调⽤的multibox_prior函数的sizes参数传递两个⽐例值的列表。
    anchors = test_anchor.multibox_prior(Y, sizes=size, ratios=ratio)
    #      在当前尺度下根据Y⽣成的锚框；
    cls_preds = cls_predictor(Y)
    #      预测的这些锚框的类别和偏移量（基于Y）。
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

if __name__ == '__main__':
    # 在下⾯，0.2和1.05之间的区间被均匀分成五个部分，以确定五个模块的在不同尺度下的较⼩值：
    #    0.2、0.37、0.54、0.71和0.88。
    # 之后，他们较⼤的值由sqrt(0.2 * 0.37) = 0.272、sqrt(0.37 * 0.54) = 0.447等给出。
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    print("num_anchors : ", num_anchors)

# 现在，我们就可以按如下⽅式定义完整的模型TinySSD了。
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                    num_anchors))
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        # 我们将预测结果转成⼆维的（批量⼤⼩，⾼×宽×通道数）的格式，以⽅便之后在维度1上的连结。
        cls_preds = concat_preds(cls_preds)
        # 之后按照类别分开。
        cls_preds = cls_preds.reshape(
                cls_preds.shape[0], -1, self.num_classes + 1)
        # 我们将预测结果转成⼆维的（批量⼤⼩，⾼×宽×通道数）的格式，以⽅便之后在维度1上的连结。
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

if __name__ == '__main__':
    # 我们创建⼀个模型实例。
    # ⽬标的类别数为1。
    net = TinySSD(num_classes=1) 
    # 然后使⽤它对⼀个256 × 256像素的⼩批量图像X执⾏前向传播。
    # 第⼀个模块输出特征图的形状为32 × 32。
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    print('output anchors:', anchors.shape)
    # 由于以特征图的每个单元为中⼼有4个锚框⽣成，因此在所有五个尺度下，
    # 每个图像总共⽣成(322 + 162 + 82 + 42 + 1) × 4 = 5444个锚框。
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)

if __name__ == '__main__':
    # 现在，我们将描述如何训练⽤于⽬标检测的单发多框检测模型。
    # ⾸先，让我们读取⾹蕉检测数据集。
    batch_size = 32
    train_iter, _ = test_object_detection_dataset.load_data_bananas(batch_size)

    # ⾹蕉检测数据集中，⽬标的类别数为1。定义好模型后，我们需要初始化其参数并定义优化算法。
    device, net = train_framework.try_gpu(), TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    # 有关锚框类别的损失：我们可以简单地复⽤之前图像分类问题⾥⼀直使⽤的交叉熵损失函数来计算；
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    # 有关正类锚框偏移量的损失：，我们在这⾥使⽤L1范数损失，即预测值和真实值之差的绝对值。
    bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # ⽬标检测有两种类型的损失。
    # 设定第⼀种有关锚框类别的损失
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
            cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # 设定第⼆种有关正类锚框偏移量的损失
    # 掩码变量bbox_masks令负类锚框和填充锚框不参与损失的计算。
    bbox = bbox_loss(bbox_preds * bbox_masks,
            bbox_labels * bbox_masks).mean(dim=1)
    # 最后，我们将锚框类别和偏移量的损失相加，以获得模型的最终损失函数。
    return cls + bbox

# 我们可以沿⽤准确率评价分类结果。
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后⼀维，argmax需要指定最后⼀维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    # 由于偏移量使⽤了L1范数损失，我们使⽤平均绝对误差来评价边界框的预测结果。
    # 这些预测结果是从⽣成的锚框及其预测偏移量中获得的。
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

if __name__ == '__main__':
    # 训练模型
    num_epochs, timer = 20, my_timer.Timer()
    animator = train_framework.Animator(xlabel='epoch', xlim=[1, num_epochs],
                    legend=['class error', 'bbox mae'])
    net = net.to(device)
    for epoch in range(num_epochs):
        # 训练精确度的和，训练精确度的和中的⽰例数
        # 绝对误差的和，绝对误差的和中的⽰例数
        metric = train_framework.Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            # ⽣成多尺度的锚框(anchors)，为每个锚框预测类别(cls_preds)和偏移量(bbox_preds)
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = test_anchor.multibox_target(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                        bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on ' 
        f'{str(device)}')

if __name__ == '__main__':
    # 在预测阶段，我们希望能把图像⾥⾯所有我们感兴趣的⽬标检测出来。
    # 在下⾯，我们读取并调整测试图像的⼤⼩，然后将其转成卷积层需要的四维格式。
    X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # 使⽤下⾯的multibox_detection函数，我们可以根据锚框及其预测偏移量得到预测边界框。
    output = test_anchor.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]
if __name__ == '__main__':
    output = predict(X)

def display(img, output, threshold):
    my_plt.set_figsize((5, 5))
    fig = my_plt.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        my_plt.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
if __name__ == '__main__':
    # 最后，我们筛选所有置信度不低于0.9的边界框，做为最终输出。
    display(img, output.cpu(), threshold=0.9)





