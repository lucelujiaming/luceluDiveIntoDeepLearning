import os
import torch
import torchvision

import my_download
import my_plt
import train_framework

# if __name__ == '__main__':
# 最重要的语义分割数据集之⼀是Pascal VOC2012(181)。
# 数据集的tar⽂件⼤约为2GB，所以下载可能需要⼀段时间。
#@save
my_download.DATA_HUB['voc2012'] = (my_download.DATA_URL + 
	'VOCtrainval_11-May-2012.tar', '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
voc_dir = my_download.download_extract('voc2012', 'VOCdevkit/VOC2012')

# 将所有输⼊的图像和标签读⼊内存。
#@save
def read_voc_images(voc_dir, is_train=True):
	"""读取所有VOC图像并标注"""
	# 进⼊路径../data/VOCdevkit/VOC2012之后，我们可以看到数据集的不同组件。
	# ImageSets/Segmentation路径包含⽤于训练和测试样本的⽂本⽂件，
	txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 
					'train.txt' if is_train else 'val.txt')
	mode = torchvision.io.image.ImageReadMode.RGB
	with open(txt_fname, 'r') as f:
		images = f.read().split()
	features, labels = [], []
	for i, fname in enumerate(images):
		# ⽽JPEGImages路径存储着每个⽰例的输⼊图像。
		features.append(torchvision.io.read_image(os.path.join(
				voc_dir, 'JPEGImages', f'{fname}.jpg')))
		# ⽽SegmentationClass路径存储着每个⽰例的标签。
		# 此处的标签也采⽤图像格式，其尺⼨和它所标注的输⼊图像的尺⼨相同。
		labels.append(torchvision.io.read_image(os.path.join(
				voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
	return features, labels

if __name__ == '__main__':
	# 将所有输⼊的图像和标签读⼊内存。
	train_features, train_labels = read_voc_images(voc_dir, True)

	# 下⾯我们绘制前5个输⼊图像及其标签。
	# 在标签图像中，⽩⾊和⿊⾊分别表⽰边框和背景，⽽其他颜⾊则对应不同的类别。
	n = 5
	imgs = train_features[0:n] + train_labels[0:n]
	imgs = [img.permute(1,2,0) for img in imgs]
	my_plt.show_images(imgs, 2, n);
	my_plt.plt.show()

# 接下来，我们列举RGB颜⾊值和类名。
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
				[0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
				[64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
				[64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
				[0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
				[0, 64, 128]]
#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 
		'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
		'diningtable', 'dog', 'horse', 'motorbike', 'person', 
		'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 我们定义了voc_colormap2label函数来构建从上述RGB颜⾊值到类别索引的映射，
# ⽽voc_label_indices函数将RGB值映射到在Pascal VOC2012数据集中的类别索引。
#@save
def voc_colormap2label():
	"""构建从RGB到VOC类别索引的映射"""
	# 创建一个稀疏表。大小为256的3次方，也就是16777216。
	colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
	# 之后填充这个稀疏表。
	for i, colormap in enumerate(VOC_COLORMAP):		
		print("i, colormap[0], colormap[1], colormap[2] :", 
			i, colormap[0], colormap[1], colormap[2])
		# 索引为RGB颜⾊组合值，取值为这个RGB颜⾊值在VOC_COLORMAP中的索引。
		# 而这个索引也是VOC_CLASSES类别数组的索引。
		colormap2label[
			(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
	return colormap2label
#@save
def voc_label_indices(colormap, colormap2label):
	"""将VOC标签中的RGB值映射到它们的类别索引"""
	# 把标签转换为RGB颜⾊值，
	colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
	# 把RGB颜⾊值转换为RGB颜⾊组合值。
	idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
				+ colormap[:, :, 2])
	# 从稀疏表中得到类别索引
	return colormap2label[idx]
	
if __name__ == '__main__':
	# 例如，在第⼀张样本图像中，⻜机头部区域的类别索引为1，⽽背景索引为0。
	y = voc_label_indices(train_labels[0], voc_colormap2label())
	print("y = voc_label_indices() return : ", y)
	print("y.shape : ", y.shape)
	print("y[105:115, 130:140], VOC_CLASSES[1] : ", 
		y[105:115, 130:140], VOC_CLASSES[1])

# 随机裁剪图像。传入四个参数：
#    feature - 图像
#    label   - 标签
#    height, width   - 裁剪的宽和高
#@save
def voc_rand_crop(feature, label, height, width):
	"""随机裁剪特征和标签图像"""
	# 1. 将图像裁剪为固定尺⼨
	rect = torchvision.transforms.RandomCrop.get_params(
					feature, (height, width))
	# 2. 我们使⽤图像增⼴中的随机裁剪，裁剪输⼊图像的相同指定区域。
	feature = torchvision.transforms.functional.crop(feature, *rect)
	# 3. 我们使⽤图像增⼴中的随机裁剪，裁剪输⼊标签的相同指定区域。
	label = torchvision.transforms.functional.crop(label, *rect)
	return feature, label

if __name__ == '__main__':
	imgs = []
	# 随机裁剪图像。
	for _ in range(n):
		imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
	# 把批量对应的维度移到后面。
	imgs = [img.permute(1, 2, 0) for img in imgs]
	my_plt.show_images(imgs[::2] + imgs[1::2], 2, n);
	my_plt.plt.show()

# 我们通过继承⾼级API提供的Dataset类，⾃定义了⼀个语义分割数据集类VOCSegDataset。
#@save
class VOCSegDataset(torch.utils.data.Dataset):
	"""⼀个⽤于加载VOC数据集的⾃定义数据集"""
	def __init__(self, is_train, crop_size, voc_dir):
		self.transform = torchvision.transforms.Normalize(
				mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.crop_size = crop_size
		features, labels = read_voc_images(voc_dir, is_train=is_train)
		# 对输⼊图像的RGB三个通道的值分别做标准化。
		self.features = [self.normalize_image(feature)
						for feature in self.filter(features)]
		# 有些图像的尺⼨可能⼩于随机裁剪所指定的输出尺⼨，通过⾃定义的filter函数移除掉。
		self.labels = self.filter(labels)
		self.colormap2label = voc_colormap2label()
		print('read ' + str(len(self.features)) + ' examples')
	# 对输⼊图像的RGB三个通道的值分别做标准化。也就是除以255。
	def normalize_image(self, img):
		return self.transform(img.float() / 255)
	# 由于数据集中有些图像的尺⼨可能⼩于随机裁剪所指定的输出尺⼨，
	# 这些样本可以通过⾃定义的filter函数移除掉。
	def filter(self, imgs):
		return [img for img in imgs if (
			img.shape[1] >= self.crop_size[0] and
			img.shape[2] >= self.crop_size[1])]
	# 通过实现__getitem__函数，
	# 我们可以任意访问数据集中索引为idx的输⼊图像及其每个像素的类别索引。
	def __getitem__(self, idx):
		# 随机裁剪图像。
		feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
						*self.crop_size)
		# 将VOC标签中的RGB值映射到它们的类别索引
		return (feature, voc_label_indices(label, self.colormap2label))
	def __len__(self):
		return len(self.features)

if __name__ == '__main__':
	# 假设我们指定随机裁剪的输出图像的形状为320 × 480，
	crop_size = (320, 480)
	# 下⾯我们可以查看训练集和测试集所保留的样本个数。
	# 我们通过⾃定义的VOCSegDataset类来创建训练集的实例。
	voc_train = VOCSegDataset(True, crop_size, voc_dir)
	# 我们通过⾃定义的VOCSegDataset类来创建测试集的实例。
	voc_test = VOCSegDataset(False, crop_size, voc_dir)

	# 设批量⼤⼩为64，我们定义训练集的迭代器。
	batch_size = 64
	train_iter = torch.utils.data.DataLoader(
		voc_train, batch_size, shuffle=True, drop_last=True,
		num_workers=train_framework.get_dataloader_workers())
	# 打印第⼀个⼩批量的形状会发现：与图像分类或⽬标检测不同，这⾥的标签是⼀个三维数组。
	for X, Y in train_iter:
		print(X.shape)
		print(Y.shape)
		break

# 最后，我们定义以下load_data_voc函数来下载并读取Pascal VOC2012语义分割数据集。
# 它返回训练集和测试集的数据迭代器。
#@save
def load_data_voc(batch_size, crop_size):
	"""加载VOC语义分割数据集"""
	voc_dir = my_download.download_extract('voc2012', os.path.join(
						'VOCdevkit', 'VOC2012'))
	# 读取Pascal VOC2012语义分割数据集。
	num_workers = train_framework.get_dataloader_workers()
	train_iter = torch.utils.data.DataLoader(
			VOCSegDataset(True, crop_size, voc_dir), batch_size,
			shuffle=True, drop_last=True, num_workers=num_workers)
	test_iter = torch.utils.data.DataLoader(
			VOCSegDataset(False, crop_size, voc_dir), batch_size,
			drop_last=True, num_workers=num_workers)
	# 返回训练集和测试集的数据迭代器。
	return train_iter, test_iter





