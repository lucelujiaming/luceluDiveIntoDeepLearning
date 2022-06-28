import os
import torch
import torchvision
from torch import nn

import my_download
import my_plt
import train_framework

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 我们使⽤的热狗数据集来源于⽹络。
# 该数据集包含1400张热狗的“正类”图像，以及包含尽可能多的其他⻝物的“负类”图像。
#@save
my_download.DATA_HUB['hotdog'] = (my_download.DATA_URL + 'hotdog.zip', 
		'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = my_download.download_extract('hotdog')

# 我们创建两个实例来分别读取训练和测试数据集中的所有图像⽂件。
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 下⾯显⽰了前8个正类样本图⽚和最后8张负类样本图⽚。正如你所看到的，图像的⼤⼩和纵横⽐各有不同。
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
my_plt.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
my_plt.plt.show()

# 对于RGB（红、绿和蓝）颜⾊通道，我们分别标准化每个通道。
# 具体⽽⾔，该通道的每个值减去该通道的平均值，然后将结果除以该通道的标准差。
# 使⽤RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize(
			[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 在训练期间，我们⾸先从图像中裁切随机⼤⼩和随机⻓宽⽐的区域，然后将该区域缩放为224×224输⼊图像。
train_augs = torchvision.transforms.Compose([
		torchvision.transforms.RandomResizedCrop(224),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.ToTensor(),
		normalize])
# 在测试过程中，我们将图像的⾼度和宽度都缩放到256像素，然后裁剪中央224 × 224区域作为输⼊。
test_augs = torchvision.transforms.Compose([
		torchvision.transforms.Resize(256),
		torchvision.transforms.CenterCrop(224),
		torchvision.transforms.ToTensor(),
		normalize])

# 我们使⽤在ImageNet数据集上预训练的ResNet-18作为源模型。
# 在这⾥，我们指定pretrained=True以⾃动下载预训练的模型参数。
# 如果你⾸次使⽤此模型，则需要连接互联⽹才能下载。
pretrained_net = torchvision.models.resnet18(pretrained=True)
# 源模型实例包含许多特征层和⼀个输出层fc。
# 此划分的主要⽬的是促进对除输出层以外所有层的模型参数进⾏微调。
print("pretrained_net.fc :", pretrained_net.fc)

finetune_net = torchvision.models.resnet18(pretrained=True)
# ⽬标模型finetune_net中成员变量features的参数被初始化为源模型相应层的模型参数。
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);


# ⾸先，我们定义了⼀个训练函数train_fine_tuning，该函数使⽤微调，因此可以多次调⽤。
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
					param_group=True):
	train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
			os.path.join(data_dir, 'train'), transform=train_augs),
			batch_size=batch_size, shuffle=True)
	test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
			os.path.join(data_dir, 'test'), transform=test_augs),
			batch_size=batch_size)
	devices = train_framework.try_all_gpus()
	loss = nn.CrossEntropyLoss(reduction="none")
	# 如果param_group=True，输出层中的模型参数将使⽤⼗倍的学习率
	if param_group:
		params_1x = [param for name, param in net.named_parameters()
					if name not in ["fc.weight", "fc.bias"]]
		# 假设Trainer实例中的学习率为η，我们将成员变量output中参数的学习率设置为10η
		trainer = torch.optim.SGD([{'params': params_1x},
				{'params': net.fc.parameters(),
				'lr': learning_rate * 10}],
				lr=learning_rate, weight_decay=0.001)
	else:
		trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
				weight_decay=0.001)
	train_framework.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
					devices)

# 我们使⽤较⼩的学习率，通过微调预训练获得的模型参数。
train_fine_tuning(finetune_net, 5e-5)

# 为了进⾏⽐较，我们定义了⼀个相同的模型，但是将其所有模型参数初始化为随机值。
# 由于整个模型需要从头开始训练，因此我们需要使⽤更⼤的学习率。
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)


