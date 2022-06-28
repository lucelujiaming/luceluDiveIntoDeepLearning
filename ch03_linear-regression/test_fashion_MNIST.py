import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import my_plt
import my_timer

my_plt.use_svg_display()

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
	root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
	root="../data", train=False, transform=trans, download=True)
# 每个输⼊图像的⾼度和宽度均为28像素。数据集由灰度图像组成，其通道数为1。
print("len(mnist_train), len(mnist_test) : ", len(mnist_train), len(mnist_test))
print("mnist_train[0][0].shape : ", mnist_train[0][0].shape)

# Fashion-MNIST中包含的10个类别，分别为：
# t-shirt（T恤）、trouser（裤⼦）、pullover（套衫）、
# dress（连⾐裙）、coat（外套）、sandal（凉鞋）、
# shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
# 以下函数⽤于在数字标签索引及其⽂本名称之间进⾏转换。
def get_fashion_mnist_labels(labels): #@save
	"""返回Fashion-MNIST数据集的⽂本标签"""
	text_labels = ['t-shirt', 'trouser', 'pullover', 
		'dress', 'coat', 'sandal', 'shirt', 
		'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in labels]
# 创建⼀个函数来可视化这些样本。
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
	"""绘制图像列表"""
	figsize = (num_cols * scale, num_rows * scale)
	_, axes = my_plt.plt.subplots(num_rows, num_cols, figsize=figsize)
	axes = axes.flatten()
	for i, (ax, img) in enumerate(zip(axes, imgs)):
		if torch.is_tensor(img):
			# 图⽚张量
			ax.imshow(img.numpy())
		else:
			# PIL图⽚
			ax.imshow(img)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		if titles:
			ax.set_title(titles[i])
	return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, 
# 		titles=get_fashion_mnist_labels(y));
# my_plt.plt.show()

# 在每次迭代中，数据加载器每次都会读取⼀⼩批量数据，⼤⼩为batch_size。
batch_size = 256
def get_dataloader_workers(): #@save
	"""使⽤4个进程来读取数据"""
	return 4
# 通过内置数据迭代器，我们可以随机打乱了所有样本，从⽽⽆偏⻅地读取⼩批量。
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
							num_workers=get_dataloader_workers())
# 我们看⼀下读取训练数据所需的时间。
timer = my_timer.Timer()
for X, y in train_iter:
	continue
print(f'{timer.stop():.2f} sec')

# 该函数⽤于获取和读取Fashion-MNIST数据集。
# 返回训练集和验证集的数据迭代器。
def load_data_fashion_mnist(batch_size, resize=None): #@save
	"""下载Fashion-MNIST数据集，然后将其加载到内存中"""
	# ToTensor()将shape为(H, W, C)的nump.ndarray或img
	#           转为shape为(C, H, W)的tensor。
	# 之后将每一个数值归一化到[0,1]，
	# 而归一化方法也比较简单，直接除以255即可。
	trans = [transforms.ToTensor()]
	# 如果存在可选参数resize，将图像⼤⼩调整为另⼀种形状。
	# 之后个可选参数resize，⽤来将图像⼤⼩调整为另⼀种形状。
	if resize:
		trans.insert(0, transforms.Resize(resize))
	# 定义转换方式，transforms.Compose将多个转换函数组合起来使用
	trans = transforms.Compose(trans)
	# 通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中。
	mnist_train = torchvision.datasets.FashionMNIST(
		root="../data", train=True, transform=trans, download=True)
	mnist_test = torchvision.datasets.FashionMNIST(
		root="../data", train=False, transform=trans, download=True)
	# 返回训练集和验证集的数据迭代器。
	return (data.DataLoader(mnist_train, batch_size, shuffle=True,
			num_workers=get_dataloader_workers()),
		data.DataLoader(mnist_test, batch_size, shuffle=False,
		num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
	print(X.shape, X.dtype, y.shape, y.dtype)
	break



