import torch
import torchvision
from torch import nn
from torch.nn import functional as F

import PIL.Image
import my_plt
import my_download
import train_framework
import test_semantic_segmentation_and_dataset

# 下⾯我们了解⼀下全卷积⽹络模型最基本的设计。
#    全卷积⽹络先使⽤卷积神经⽹络抽取图像特征，
#    然后通过1 × 1卷积层将通道数变换为类别个数，
#    最后通过转置卷积层将特征图的⾼和宽变换为输⼊图像的尺⼨。
# 因此，模型输出与输⼊图像的⾼和宽相同，且最终输出通道包含了该空间位置像素的类别预测。

# 我们使⽤在ImageNet数据集上预训练的ResNet-18模型来提取图像特征，并将该⽹络记为pretrained_net。
pretrained_net = torchvision.models.resnet18(pretrained=True)
# ResNet-18模型的最后⼏层包括全局平均汇聚层和全连接层，然⽽全卷积⽹络中不需要它们。
print("list(pretrained_net.children())[-3:] : ", 
	list(pretrained_net.children())[-3:])

# 我们创建⼀个全卷积⽹络net。
# 它复制了ResNet-18中⼤部分的预训练层，除了最后的全局平均汇聚层和最接近输出的全连接层。
net = nn.Sequential(*list(pretrained_net.children())[:-2])
# 给定⾼度为320和宽度为480的输⼊，net的前向传播将输⼊的⾼和宽减⼩⾄原来的1/32，即10和15。 
X = torch.rand(size=(1, 3, 320, 480))
# net(X).shape :  torch.Size([1, 512, 10, 15])
print("net(X).shape : ", net(X).shape)

# 接下来，我们使⽤1 × 1卷积层将输出通道数转换为Pascal VOC2012数据集的类数（21类）。
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
# 最后，我们需要将特征图的⾼度和宽度增加32倍，从⽽将其变回输⼊图像的⾼和宽。
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
			kernel_size=64, padding=16, stride=32))

# 双线性插值的上采样可以通过转置卷积层实现，内核由以下bilinear_kernel函数构造。
# 限于篇幅，我们只给出bilinear_kernel函数的实现，不讨论算法的原理。补充如下：
# 双线性插值（Bilinear Interpolation）
#     https://blog.csdn.net/qq_38701868/article/details/103511663
# 在从低分辨率图像生成高分辨率图像的过程，用来恢复图像中所丢失的信息。
# 线性插值在数学上的几何意义为：利用过点(x0, y0)和(x1, y1)的直线L(x)来近似原函数f(x)。
# 若想要计算函数f(x)上的x处的y值，则公式如下：
#         (y - y0)/(x - x0) = (y - y1)/(x - x1)
#     ==> y = (x1 - x)/(x1 - x0) * y0 + (x1 - x)/(x1 - x0) * y1
# 双线性插值: 
# 线性插值是在一个方向上进行插值，如上例中为x轴方向，而双线性插值则是在两个方向分别进行一次线性插值。
# 如：
#      |    |         |        |
#    y2|____|_________|________|____
#      |    |Q12      |R2      |Q22
#      |    |         |        |
#     y|____|_________|________|____
#      |    |         |P       |
#      |    |         |        |
#      |    |         |        |
#      |    |         |        |
#    y1|____|_________|________|____
#      |    |Q11      |R1      |Q21
#      |____|_________|________|____
#           x1        x        x2
# 
# 假设我们已知函数 f 在 Q11 = (x1, y1), Q12 = (x1, y2), Q21 = (x2, y1)以及 Q22 = (x2, y2)四个点的值。
# 最常见的情况，f就是一个像素点的像素值。首先在 x 方向进行线性插值，得到 : 
#     f(R1) ≈ (x2 - x)/(x2 - x1) * f(Q11) + (x - x1)/(x2 - x1) * f(Q21) where R1 = (x, y1)
#     f(R2) ≈ (x2 - x)/(x2 - x1) * f(Q12) + (x - x1)/(x2 - x1) * f(Q22) where R2 = (x, y2)
# 然后在 y 方向进行线性插值，得到: 
#     f(P) ≈ (y2 - y)/(y2 - y1) * f(R1) + (y - y1)/(y2 - y1) * f(R2) 
# 即f(x,y)为：
#     f(x,y) = f(Q11)*(x2−x)*(y2−y) + f(Q21)*(x−x1)*(y2−y) 
#            + f(Q12)*(x2−x)*(y−y1) + f(Q22)*(x−x1)*(y−y1)
# 由于图像双线性插值只会用相邻的4个点，因此上述公式的分母都是1。
# 如四个已知点坐标分别为(0,0)、(0,1)、(1,0)和(1,1)，那么插值公式就可以化简为：
#     f(x,y) ≈ f(0,0)*(1-x)*(1-y) + f(1,0)*x*(1-y) + f(0,1)*(1-x)*y + f(1,1)*x*y
# 线性插值的结果与插值的顺序无关。先进行y方向的插值，然后进行x方向的插值，所得到的结果是一样的。

def bilinear_kernel(in_channels, out_channels, kernel_size):
	# 根据核的大小计算缩放比例。
	# 假设我们要将 3×6 的图象转变为 6×12 的图象，即已知 upscale_factor = 2。
	factor = (kernel_size + 1) // 2
	# Centre location of the filter for which value is calculated
	# filter_shape 是卷积核的形状，由 kernel_size = 2 * upscale_factor - upscale_factor % 2 计算得到，
	#     filter_shape=[4,4,1,1]。
	# filter_shape is [width, height, num_in_channels, num_out_channels]
	if kernel_size % 2 == 1:
		center = factor - 1
	else:
		center = factor - 0.5
	og = (torch.arange(kernel_size).reshape(-1, 1),
			torch.arange(kernel_size).reshape(1, -1))
	filt = (1 - torch.abs(og[0] - center) / factor) * \
		 (1 - torch.abs(og[1] - center) / factor)
	weight = torch.zeros((in_channels, out_channels,
			kernel_size, kernel_size))
	weight[range(in_channels), range(out_channels), :, :] = filt
	return weight

# 我们构造⼀个将输⼊的⾼和宽放⼤2倍的转置卷积层，
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
							bias=False)
# 并将其卷积核⽤bilinear_kernel函数初始化。
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
# 读取图像X，将上采样的结果记作Y。为了打印图像，我们需要调整通道维的位置。
img = torchvision.transforms.ToTensor()(PIL.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
# 可以看到，转置卷积层将图像的⾼和宽分别放⼤了2倍。
# 除了坐标刻度不同，双线性插值放⼤的图像和在13.3节中打印出的原图看上去没什么两样。
my_plt.set_figsize()
# input image shape: torch.Size([561, 728, 3])
print('input image shape:', img.permute(1, 2, 0).shape)
# my_plt.plt.imshow(img.permute(1, 2, 0));
# output image shape: torch.Size([1122, 1456, 3])
print('output image shape:', out_img.shape)
# my_plt.plt.imshow(out_img);
# 上面两张图片依次显示根本看不出来效果。这里改成并列显示。
imgs = []
imgs.append(img.permute(1, 2, 0))
imgs.append(out_img)
my_plt.show_images(imgs, 1, 2);
my_plt.plt.show()

# 在全卷积⽹络中，我们⽤双线性插值的上采样初始化转置卷积层。
W = bilinear_kernel(num_classes, num_classes, 64)
# 对于1 × 1卷积层，我们使⽤Xavier初始化参数。
net.transpose_conv.weight.data.copy_(W);

# 我们⽤ 13.9节中介绍的语义分割读取数据集。
# 指定随机裁剪的输出图像的形状为320 × 480：⾼和宽都可以被32整除。
batch_size, crop_size = 32, (320, 480)
# 下载并读取Pascal VOC2012语义分割数据集。返回训练集和测试集的数据迭代器。
train_iter, test_iter = test_semantic_segmentation_and_dataset.load_data_voc(batch_size, crop_size)

# 现在我们可以训练全卷积⽹络了。
def loss(inputs, targets):
	return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
num_epochs, lr, wd, devices = 5, 0.001, 1e-3, train_framework.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
train_framework.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 在预测时，
def predict(img):
	# 我们需要将输⼊图像在各个通道做标准化，并转成卷积神经⽹络所需要的四维输⼊格式。
	X = test_iter.dataset.normalize_image(img).unsqueeze(0)
	pred = net(X.to(devices[0])).argmax(dim=1)
	return pred.reshape(pred.shape[1], pred.shape[2])
# 为了可视化预测的类别给每个像素，我们将预测类别映射回它们在数据集中的标注颜⾊。
def label2image(pred):
	colormap = torch.tensor(test_semantic_segmentation_and_dataset.VOC_COLORMAP, device=devices[0])
	X = pred.long()
	return colormap[X, :]

voc_dir = my_download.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = \
	test_semantic_segmentation_and_dataset.read_voc_images(voc_dir, False)
# 为简单起⻅，我们只读取⼏张较⼤的测试图像，
n, imgs = 4, []
for i in range(n):
	# 并从图像的左上⻆开始截取形状为320×480的区域⽤于预测。
	crop_rect = (0, 0, 320, 480) 
	X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
	# 对于这些测试图像，我们逐⼀打印它们截取的区域，再打印预测结果，最后打印标注的类别。
	pred = label2image(predict(X))
	imgs += [X.permute(1,2,0), pred.cpu(),
		torchvision.transforms.functional.crop(
			test_labels[i], *crop_rect).permute(1,2,0)]
my_plt.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
my_plt.plt.show()





