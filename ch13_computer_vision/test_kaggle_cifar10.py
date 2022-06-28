# # Importing PlaidML. Make sure you follow this order
# import plaidml.keras
# plaidml.keras.install_backend()
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn

import my_timer
import my_download
import train_framework

if __name__ == '__main__':
    #@save
    my_download.DATA_HUB['cifar10_tiny'] = (my_download.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                    '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
    # 如果你使用完整的Kaggle竞赛的数据集，设置`demo`为 False
    demo = True
    if demo:
        data_dir = my_download.download_extract('cifar10_tiny')
    else:
        data_dir = '../data/cifar-10/'

# 我们需要整理数据集来训练和测试模型。
# ⾸先，我们⽤以下函数读取CSV⽂件中的标签，它返回⼀个字典，
# 该字典将⽂件名中不带扩展名的部分映射到其标签。
#@save
def read_csv_labels(fname):
    """读取fname来给标签字典返回⼀个⽂件名"""
    with open(fname, 'r') as f:
        # 跳过⽂件头⾏(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

if __name__ == '__main__':
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    print('# 训练样本 :', len(labels))
    print('# 类别 :', len(set(labels.values())))

#@save
def copyfile(filename, target_dir):
    """将⽂件复制到⽬标⽬录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

# 接下来，我们定义reorg_train_valid函数来将验证集从原始的训练集中拆分出来。
# 此函数中的参数valid_ratio是验证集中的样本数与原始训练集中的样本数之⽐。
def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1] # 验证集中每个类别的样本数
    # 令n等于样本最少的类别中的图像数量，⽽r是⽐率。
    # 验证集将为每个类别拆分出max(⌊nr⌋, 1)张图像。
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        # 让我们以valid_ratio=0.1为例，由于原始的训练集有50000张图像，
        # 因此train_valid_test/train路径中将有45000张图像⽤于训练，
        # ⽽剩下5000张图像将作为路径train_valid_test/valid中的验证集。
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label

# 下⾯的reorg_test函数⽤来在预测期间整理测试集，以⽅便读取。
#@save
def reorg_test(data_dir):
    """在预测期间整理测试集，以⽅便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
            os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

# 最后，我们使⽤⼀个函数来调⽤前⾯定义的函数read_csv_labels、reorg_train_valid和reorg_test。
def reorg_cifar10_data(data_dir, valid_ratio):
    # 整理数据集来训练和测试模型。
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    # 将验证集从原始的训练集中拆分出来。
    reorg_train_valid(data_dir, labels, valid_ratio)
    # 在预测期间整理测试集，以⽅便读取。
    reorg_test(data_dir)

if __name__ == '__main__':
    # 在这⾥，我们只将样本数据集的批量⼤⼩设置为32。
    # 在实际训练和测试中，应该使⽤Kaggle竞赛的完整数据集，并将batch_size设置为更⼤的整数，例如128。
    batch_size = 32 if demo else 128
    # 我们将10％的训练样本作为调整超参数的验证集。
    valid_ratio = 0.1
    reorg_cifar10_data(data_dir, valid_ratio)

    # 我们使⽤图像增⼴来解决过拟合的问题。例如在训练中，我们可以随机⽔平翻转图像。
    # 我们还可以对彩⾊图像的三个RGB通道执⾏标准化。下⾯，我们列出了其中⼀些可以调整的操作。
    transform_train = torchvision.transforms.Compose([
        # 在⾼度和宽度上将图像放⼤到40像素的正⽅形
        torchvision.transforms.Resize(40),
        # 随机裁剪出⼀个⾼度和宽度均为40像素的正⽅形图像，
        # ⽣成⼀个⾯积为原始图像⾯积0.64到1倍的⼩正⽅形，
        # 然后将其缩放为⾼度和宽度均为32像素的正⽅形
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                    ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])

    # 在测试期间，我们只对图像执⾏标准化，以消除评估结果中的随机性。
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])


    # 接下来，我们读取由原始图像组成的数据集，每个样本都包括⼀张图⽚和⼀个标签。
    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]

    # 在训练期间，我们需要指定上⾯定义的所有图像增⼴操作。
    train_iter, train_valid_iter = [torch.utils.data.DataLoader(
                    dataset, batch_size, shuffle=True, drop_last=True)
                    for dataset in (train_ds, train_valid_ds)]
    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                    drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                    drop_last=False)

# 我们定义了 7.6节中描述的Resnet-18模型。
def get_net():
    num_classes = 10
    net = train_framework.resnet18(num_classes, 3)
    return net
if __name__ == '__main__':
    loss = nn.CrossEntropyLoss(reduction="none")

# 我们将根据模型在验证集上的表现来选择模型并调整超参数。下⾯我们定义了模型训练函数train。
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
                    lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                            weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), my_timer.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = train_framework.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = train_framework.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_framework.train_batch_ch13(net, features, labels,
                            loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                        (metric[0] / metric[2], metric[1] / metric[2],
                        None))
        if valid_iter is not None:
            valid_acc = train_framework.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, ' 
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}' 
                     f' examples/sec on {str(devices)}')

if __name__ == '__main__':
    # 现在，我们可以训练和验证模型了，⽽以下所有超参数都可以调整。
    devices, num_epochs, lr, wd = train_framework.try_all_gpus(), 20, 2e-4, 5e-4
    lr_period, lr_decay, net = 4, 0.9, get_net()
    train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
                lr_decay)

    # 在获得具有超参数的满意的模型后，我们使⽤所有标记的数据（包括验证集）来重新训练模型并对测试集进⾏分类。
    net, preds = get_net(), []
    train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
                    lr_decay)
    for X, _ in test_iter:
        y_hat = net(X.to(devices[0]))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    sorted_ids = list(range(1, len(test_ds) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
    df.to_csv('submission.csv', index=False)







