import torch
import torchvision
from torch import nn

import PIL.Image
import my_plt

# 在对常⽤图像增⼴⽅法的探索时，我们将使⽤下⾯这个尺⼨为400 × 500的图像作为⽰例。
my_plt.set_figsize()
img = PIL.Image.open('../img/cat1.jpg')
my_plt.plt.imshow(img);
my_plt.plt.show()

# 为了便于观察图像增⼴的效果，我们下⾯定义辅助函数apply。
# 此函数在输⼊图像img上多次运⾏图像增⼴⽅法aug并显⽰所有结果。
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
	Y = [aug(img) for _ in range(num_rows * num_cols)]
	my_plt.show_images(Y, num_rows, num_cols, scale=scale)
	my_plt.plt.show()

# 左右翻转图像通常不会改变对象的类别。这是最早且最⼴泛使⽤的图像增⼴⽅法之⼀。
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 上下翻转图像不如左右图像翻转那样常⽤。但是，⾄少对于这个⽰例图像，上下翻转不会妨碍识别。
apply(img, torchvision.transforms.RandomVerticalFlip())

# 我们可以通过对图像进⾏随机裁剪，使物体以不同的⽐例出现在图像的不同位置。
# 我们随机裁剪⼀个⾯积为原始⾯积10%到100%的区域，
# 该区域的宽⾼⽐从0.5到2之间随机取值。然后，区域的宽度和⾼度都被缩放到200像素。
shape_aug = torchvision.transforms.RandomResizedCrop(
				(200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 另⼀种增⼴⽅法是改变颜⾊。我们可以改变图像颜⾊的四个⽅⾯：亮度、对⽐度、饱和度和⾊调。
# 在下⾯的⽰例中，我们随机更改图像的亮度，
# 随机值为原始图像的50%（1 − 0.5）到150%（1 + 0.5）之间。
apply(img, torchvision.transforms.ColorJitter(
		brightness=0.5, contrast=0, saturation=0, hue=0))
# 同样，我们可以随机更改图像的⾊调。
apply(img, torchvision.transforms.ColorJitter(
		brightness=0, contrast=0, saturation=0, hue=0.5))

# 我们还可以创建⼀个RandomColorJitter实例，并设置如何同时随机更改
# 图像的亮度（brightness）、对⽐度（contrast）、饱和度（saturation）和⾊调（hue）。
color_aug = torchvision.transforms.ColorJitter(
		brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 在实践中，我们将结合多种图像增⼴⽅法。
augs = torchvision.transforms.Compose([
		torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)



