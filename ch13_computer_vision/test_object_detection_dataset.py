import os
import pandas as pd
import torch
import torchvision

import my_download
import my_plt
import test_anchor

# 为了快速测试⽬标检测模型，我们收集并标记了⼀个⼩型数据集。
# ⾸先，我们拍摄了⼀组⾹蕉的照⽚，并⽣成了1000张不同⻆度和⼤⼩的⾹蕉图像。
# 然后，我们在⼀些背景图⽚的随机位置上放⼀张⾹蕉的图像。
# 最后，我们在图⽚上为这些⾹蕉标记了边界框。
# 包含所有图像和CSV标签⽂件的⾹蕉检测数据集可以直接从互联⽹下载。
#@save
my_download.DATA_HUB['banana-detection'] = (
	my_download.DATA_URL + 'banana-detection.zip', 
	'5de26c8fce5ccdea9f91267273464dc968d20d72')

#@save
def read_data_bananas(is_train=True):
	"""读取⾹蕉检测数据集中的图像和标签"""
	# 我们读取⾹蕉检测数据集。
	data_dir = my_download.download_extract('banana-detection')
	# 该数据集包括⼀个的CSV⽂件，内含⽬标类别标签和位于左上⻆和右下⻆的真实边界框坐标。
	csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
							else 'bananas_val', 'label.csv')
	csv_data = pd.read_csv(csv_fname)
	csv_data = csv_data.set_index('img_name')
	images, targets = [], []
	for img_name, target in csv_data.iterrows():
		images.append(torchvision.io.read_image(
		os.path.join(data_dir, 'bananas_train' if is_train else
						'bananas_val', 'images', f'{img_name}')))
		# 这⾥的target包含（类别，左上⻆x，左上⻆y，右下⻆x，右下⻆y），
		# 其中所有图像都具有相同的⾹蕉类（索引为0）
		targets.append(list(target))
	return images, torch.tensor(targets).unsqueeze(1) / 256

#@save
class BananasDataset(torch.utils.data.Dataset):
	"""⼀个⽤于加载⾹蕉检测数据集的⾃定义数据集"""
	def __init__(self, is_train):
		# 使⽤read_data_bananas函数读取图像和标签，
		self.features, self.labels = read_data_bananas(is_train)
		print('read ' + str(len(self.features)) + (f' training examples' if
			is_train else f' validation examples'))
	def __getitem__(self, idx):
		return (self.features[idx].float(), self.labels[idx])
	def __len__(self):
		return len(self.features)

#@save
def load_data_bananas(batch_size):
	"""加载⾹蕉检测数据集"""
	# 为训练集和测试集返回两个数据加载器实例。
	# 对于测试集，⽆须按随机顺序读取它。
	train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
						batch_size, shuffle=True)
	val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
						batch_size)
	return train_iter, val_iter

if __name__ == '__main__':
	# 让我们读取⼀个⼩批量，并打印其中的图像和标签的形状。
	batch_size, edge_size = 32, 256
	train_iter, _ = load_data_bananas(batch_size)
	batch = next(iter(train_iter))
	# 图像的⼩批量的形状为（批量⼤⼩、通道数、⾼度、宽度）。
	# batch[0].shape :  torch.Size([32, 3, 256, 256])
	# 图像的⼩批量的形状为（批量⼤⼩、通道数、⾼度、宽度），它与我们之前图像分类任务中的相同。
	print("batch[0].shape : ", batch[0].shape)
	# batch[1].shape :  torch.Size([32, 1, 5])
	# 标签的⼩批量的形状为（批量⼤⼩，m， 5），其中m是数据集的任何图像中边界框可能出现的最⼤数量。
	print("batch[1].shape : ", batch[1].shape)

	# 让我们展⽰10幅带有真实边界框的图像。
	# 我们可以看到在所有这些图像中⾹蕉的旋转⻆度、⼤⼩和位置都有所不同。
	imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
	axes = my_plt.show_images(imgs, 2, 5, scale=2)
	for ax, label in zip(axes, batch[1][0:10]):
		test_anchor.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
	my_plt.plt.show()


