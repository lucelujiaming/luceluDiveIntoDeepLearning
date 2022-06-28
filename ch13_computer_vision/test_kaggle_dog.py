import os
import torch
import torchvision
from torch import nn

import my_timer
import my_download
import train_framework
import test_kaggle_cifar10

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 这个数据集实际上是著名的ImageNet的数据集⼦集。
# ⽐赛数据集分为训练集和测试集，分别包含RGB（彩⾊）通道的10222张、10357张JPEG图像。
# 在训练数据集中，有120种⽝类。
# 其中⽂件夹train/和test/分别包含训练和测试狗图像，labels.csv包含训练图像的标签。
#@save
my_download.DATA_HUB['dog_tiny'] = (my_download.DATA_URL + 'kaggle_dog_tiny.zip', 
          '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d') 
# 如果你使⽤Kaggle⽐赛的完整数据集，请将下⾯的变量更改为False
demo = True
if demo:
    data_dir = my_download.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')

# 下⾯的reorg_dog_data函数读取训练数据标签、拆分验证集并整理训练集。
def reorg_dog_data(data_dir, valid_ratio):
    labels = test_kaggle_cifar10.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    test_kaggle_cifar10.reorg_train_valid(data_dir, labels, valid_ratio)
    test_kaggle_cifar10.reorg_test(data_dir)
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

# 下⾯我们看⼀下如何在相对较⼤的图像上使⽤图像增⼴。
transform_train = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始⾯积的0.08到1之间，⾼宽⽐在3/4和4/3之间。
    # 然后，缩放图像以创建224x224的新图像
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                     ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对⽐度和饱和度
    torchvision.transforms.ColorJitter(brightness=0.4,
                                    contrast=0.4,
                                    saturation=0.4),
    # 添加随机噪声
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
# 测试时，我们只使⽤确定性的图像预处理操作。
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # 从图像中⼼裁切224x224⼤⼩的图⽚
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

# 我们可以读取整理后的含原始图像⽂件的数据集。
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]
# 下⾯我们创建数据加载器实例# 。
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)
        for dataset in (train_ds, train_valid_ds)]
valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                    drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                    drop_last=False)

def get_net(devices):
    # 同样，本次⽐赛的数据集是ImageNet数据集的⼦集。
    # 因此，我们可以使⽤13.2节中讨论的⽅法在完整ImageNet数据集上选择预训练的模型，
    # 然后使⽤该模型提取图像特征，以便将其输⼊到定制的⼩规模输出⽹络中。
    finetune_net = nn.Sequential()
    # 在这⾥，我们选择预训练的ResNet-34模型，我们只需重复使⽤此模型的输出层（即提取的特征）的输⼊。
    finetune_net.features = torchvision.models.resnet34(pretrained=True) 
    # 然后，我们可以⽤⼀个可以训练的⼩型⾃定义输出⽹络替换原始输出层，例如堆叠两个完全连接的图层。
    # 定义⼀个新的输出⽹络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                            nn.ReLU(),
                            nn.Linear(256, 120))
    # 将模型参数分配给⽤于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    以下内容不重新训练⽤于特征提取的预训练模型，这节省了梯度下降的时间和内存空间。
    # 冻结参数。
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

# 在计算损失之前，我们⾸先获取预训练模型的输出层的输⼊，即提取的特征。
# 然后我们使⽤此特征作为我们⼩型⾃定义输出⽹络的输⼊来计算损失。
loss = nn.CrossEntropyLoss(reduction='none')
def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum / n).to('cpu')

# 我们将根据模型在验证集上的表现选择模型并调整超参数。
# 模型训练函数train只迭代⼩型⾃定义输出⽹络的参数。
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
                            lr_decay):
    # 只训练⼩型⾃定义输出⽹络
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                            if param.requires_grad), lr=lr,
                            momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), my_timer.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = train_framework.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)

    for epoch in range(num_epochs):
        metric = train_framework.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                            animator.add(epoch + (i + 1) / num_batches,
                            (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
            measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}' 
        f' examples/sec on {str(devices)}')

# 现在我们可以训练和验证模型了，以下超参数都是可调的。例如，我们可以增加迭代轮数。
devices, num_epochs, lr, wd = train_framework.try_all_gpus(), 2, 1e-4, 1e-4
# 另外，由于lr_period和lr_decay分别设置为2和0.9，
# 因此优化算法的学习速率将在每2个迭代后乘以0.9。
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
                lr_decay)

# 最终所有标记的数据（包括验证集）都⽤于训练模型和对测试集进⾏分类。
# 我们将使⽤训练好的⾃定义输出⽹络进⾏分类。
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
                lr_decay)
preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=0)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
        os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
# 代码将⽣成⼀个submission.csv⽂件，以 4.10节中描述的⽅式提在Kaggle上提交。
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
                    [str(num) for num in output]) + '\n')








