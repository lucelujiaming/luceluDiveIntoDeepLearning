import torch
import torchvision
from torch import nn

import PIL.Image
import my_plt
import train_framework

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ⻛格迁移（style transfer）[Gatys et al., 2016]需要两张输⼊图像：
# ⼀张是内容图像，另⼀张是⻛格图像。我们将使⽤神经⽹络修改内容图像，使其在⻛格上接近⻛格图像。
# 基于卷积神经⽹络的⻛格迁移⽅法。
#    1. ⾸先，我们初始化合成图像，例如将其初始化为内容图像。
#       该合成图像是⻛格迁移过程中唯⼀需要更新的变量，即⻛格迁移所需迭代的模型参数。
#    2. 然后，我们选择⼀个预训练的卷积神经⽹络来抽取图像的特征，其中的模型参数在训练中⽆须更新。
#       这个深度卷积神经⽹络凭借多个层逐级抽取图像的特征，我们可以选择其中某些层的输出作为内容特征或⻛格特征。
#    3. 接下来，我们通过前向传播（实线箭头⽅向）计算⻛格迁移的损失函数，
#       并通过反向传播（虚线箭头⽅向）迭代模型参数，即不断更新合成图像。
#       ⻛格迁移常⽤的损失函数由3部分组成：
#          （i）  内容损失使合成图像与内容图像在内容特征上接近；
#          （ii） ⻛格损失使合成图像与⻛格图像在⻛格特征上接近；
#          （iii）全变分损失则有助于减少合成图像中的噪点。
#    4. 最后，当模型训练结束时，我们输出⻛格迁移的模型参数，即得到最终的合成图像。

# ⾸先，我们读取内容和⻛格图像。
my_plt.set_figsize()
content_img = PIL.Image.open('../img/rainier.jpg')
my_plt.plt.imshow(content_img);
style_img = PIL.Image.open('../img/autumn-oak.jpg')
my_plt.plt.imshow(style_img);

# 下⾯，定义图像的预处理函数和后处理函数。
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])
# 预处理函数preprocess对输⼊图像在RGB三个通道分别做标准化，并将结果变换成卷积神经⽹络接受的输⼊格式。
def preprocess(img, image_shape):
	transforms = torchvision.transforms.Compose([
		torchvision.transforms.Resize(image_shape),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
	return transforms(img).unsqueeze(0)
# 后处理函数postprocess则将输出图像中的像素值还原回标准化之前的值。
def postprocess(img):
	img = img[0].to(rgb_std.device)
	# 由于图像打印函数要求每个像素的浮点数值在0到1之间，我们对⼩于0和⼤于1的值分别取0和1。
	img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
	return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# 我们使⽤基于ImageNet数据集预训练的VGG-19模型来抽取图像特征 [Gatys et al., 2016]。
pretrained_net = torchvision.models.vgg19(pretrained=True)

# VGG⽹络使⽤了5个卷积块。
# 选择原则：
#    1. 越靠近输⼊层，越容易抽取图像的细节信息；
#       反之，越靠近输出层，则越容易抽取图像的全局信息。
#    2. 为了避免过多内容图像的细节，我们选择VGG较靠近输出的层，即内容层，来输出图像的内容特征。
#       也就是说，内容层抽取图像的全局信息。关注图像画的是啥。而不关心图像是怎么画的。
#    3. 从VGG中选择不同层的输出来匹配局部和全局的⻛格，这些图层也称为⻛格层。
#       也就是说，⻛格层抽取图像的细节信息。关注图像是怎么画的。而不关心图像画的是啥。
# 因此，我们选择第四卷积块的最后⼀个卷积层作为内容层，选择每个卷积块的第⼀个卷积层作为⻛格层。
# 这些层的索引可以通过打印pretrained_net实例获取。
style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# 使⽤VGG层抽取特征时，我们只需要⽤到从输⼊层到最靠近输出层的内容层或⻛格层之间的所有层。
# 下⾯构建⼀个新的⽹络net，它只保留需要⽤到的VGG的所有层。
net = nn.Sequential(*[pretrained_net.features[i] for i in
				range(max(content_layers + style_layers) + 1)])

def extract_features(X, content_layers, style_layers):
	contents = []
	styles = []
	# 由于我们还需要中间层的输出，因此这⾥我们逐层计算，
	for i in range(len(net)):
		X = net[i](X)
		# 获得内容层和⻛格层的输出。
		if i in style_layers:
			styles.append(X)
		if i in content_layers:
			contents.append(X)
	return contents, styles

# get_contents函数对内容图像抽取内容特征；
def get_contents(image_shape, device):
	content_X = preprocess(content_img, image_shape).to(device)
	contents_Y, _ = extract_features(content_X, content_layers, style_layers)
	return content_X, contents_Y
# get_styles函数对⻛格图像抽取⻛格特征。
def get_styles(image_shape, device):
	style_X = preprocess(style_img, image_shape).to(device)
	_, styles_Y = extract_features(style_X, content_layers, style_layers)
	return style_X, styles_Y

# 下⾯我们来描述⻛格迁移的损失函数。它由内容损失、⻛格损失和全变分损失3部分组成。
# 1. 内容损失
#    平⽅误差函数的两个输⼊均为extract_features函数计算所得到的内容层的输出。
def content_loss(Y_hat, Y):
	# 与线性回归中的损失函数类似，内容损失通过平⽅误差函数衡量合成图像与内容图像在内容特征上的差异。
	# 我们从动态计算梯度的树中分离⽬标：
	# 这是⼀个规定的值，⽽不是⼀个变量。
	return torch.square(Y_hat - Y.detach()).mean()

# 格拉姆矩阵：
# 概念：n维欧式空间中任意k个向量之间两两的内积所组成的矩阵，称为这k个向量的gram矩阵。
# 如何理解Gram矩阵：度量各个维度自己的特性以及各个维度之间的关系。
# 进一步解释：从向量点乘角度有助于理解格拉姆矩阵。向量点乘可以看作衡量两个向量的相似程度。
#   对于二维向量来说，两个单位向量，方向一致，点乘为1，相互垂直，点乘为0，方向相反，点乘为-1。
#   因此在单位向量的情况下，结果由两个向量夹角cos的值决定。
#   而对于多维向量，向量点乘就是对应位置乘积之后相加，得到的结果仍然是标量，含义和二维向量一样。
#   格拉姆矩阵就是由两两向量内积组成，因此可以得出格拉姆矩阵可以度量各个维度自己的特性以及各个维度之间的关系的结论。
# 
# 下⾯定义的gram函数将格拉姆矩阵除以了矩阵中元素的个数，即chw。
def gram(X):
	# X.shape: [样本数, 通道数, ⾼, 宽]
	# 因此上，X.shape[1]为通道数。
	num_channels, n = X.shape[1], X.numel() // X.shape[1] 
	# 将此输出转换为矩阵X，其有c⾏和hw列。
	# 这个矩阵可以被看作是由c个⻓度为hw的向量x1, ..., xc组合⽽成的。
	# 其中向量xi代表了通道i上的⻛格特征。
	X = X.reshape((num_channels, n))
	# n维欧式空间中任意k个向量之间两两的内积所组成的矩阵，称为这k个向量的gram矩阵。
	# i⾏j列的元素xij即向量xi和xj的内积。它表达了通道i和通道j上⻛格特征的相关性。
	# 我们⽤这样的格拉姆矩阵来表达⻛格层输出的⻛格。
	# 这里，除以了矩阵中元素的个数，即(num_channels * n)。
	return torch.matmul(X, X.T) / (num_channels * n)
# ⾃然地，⻛格损失的平⽅误差函数的两个格拉姆矩阵输⼊分别基于合成图像与⻛格图像的⻛格层输出。
# 这⾥假设基于⻛格图像的格拉姆矩阵gram_Y已经预先计算好了。
def style_loss(Y_hat, gram_Y):
	return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 全变分损失
# 有时候，我们学到的合成图像⾥⾯有⼤量⾼频噪点，即有特别亮或者特别暗的颗粒像素。
# ⼀种常⻅的去噪⽅法是全变分去噪，降低全变分损失能够尽可能使邻近的像素值相似。
def tv_loss(Y_hat):
	# 基于公式：(13.12.1)
	return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
		torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

# 损失函数
# ⻛格转移的损失函数是内容损失、⻛格损失和总变化损失的加权和。
# 通过调节这些权重超参数，我们可以权衡合成图像在保留内容、迁移⻛格以及去噪三⽅⾯的相对重要性。
content_weight, style_weight, tv_weight = 1, 1e3, 10
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
	# 使用权重超参数计算内容损失
	contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
					contents_Y_hat, contents_Y)]
	# 使用权重超参数计算⻛格损失
	styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
					styles_Y_hat, styles_Y_gram)]
	# 使用权重超参数计算全变分损失
	tv_l = tv_loss(X) * tv_weight
	# 对所有损失求和
	l = sum(10 * styles_l + contents_l + [tv_l])
	return contents_l, styles_l, tv_l, l

# 定义⼀个简单的模型SynthesizedImage，并将合成的图像视为模型参数。
# 模型的前向传播只需返回模型参数即可。
class SynthesizedImage(nn.Module):
	def __init__(self, img_shape, **kwargs):
		super(SynthesizedImage, self).__init__(**kwargs)
		self.weight = nn.Parameter(torch.rand(*img_shape))
	def forward(self):
		return self.weight

# 下⾯，我们定义get_inits函数。
def get_inits(X, device, lr, styles_Y):
	# 该函数创建了合成图像的模型实例，并将其初始化为图像X。
	gen_img = SynthesizedImage(X.shape).to(device)
	gen_img.weight.data.copy_(X.data)
	trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
	# ⻛格图像在各个⻛格层的格拉姆矩阵styles_Y_gram将在训练前预先计算好。
	styles_Y_gram = [gram(Y) for Y in styles_Y]
	return gen_img(), styles_Y_gram, trainer

# 在训练模型进⾏⻛格迁移时，我们不断抽取合成图像的内容特征和⻛格特征，然后计算损失函数。
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
	X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
	scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
	animator = train_framework.Animator(xlabel='epoch', ylabel='loss',
						xlim=[10, num_epochs],
						legend=['content', 'style', 'TV'],
						# 内部调用subplots创建两列。第一列保留为曲线图。
						# 第二列通过调用animator.axes[1]进行绘制。
						ncols=2, figsize=(7, 2.5))
	for epoch in range(num_epochs):
		trainer.zero_grad()
		# 提取内容特征和风格特征。
		contents_Y_hat, styles_Y_hat = extract_features(
				X, content_layers, style_layers)
		# 计算内容损失、⻛格损失和总变化损失。
		contents_l, styles_l, tv_l, l = compute_loss(
				X, contents_Y_hat, styles_Y_hat, 
				contents_Y, styles_Y_gram)
		l.backward()
		trainer.step()
		scheduler.step()
		if (epoch + 1) % 10 == 0:
			# 绘制当前效果图。
			animator.axes[1].imshow(postprocess(X))
			# 绘制内容损失、⻛格损失和总变化损失的曲线。
			animator.add(epoch + 1, [float(sum(contents_l)),
				float(sum(styles_l)), float(tv_l)])
	return X

# 现在我们训练模型：
# ⾸先将内容图像和⻛格图像的⾼和宽分别调整为300和450像素，⽤内容图像来初始化合成图像。
device, image_shape = train_framework.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)



