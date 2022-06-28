import torch

import my_plt
import test_anchor

img = my_plt.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
print("h, w : ", h, w)
figPic = my_plt.plt.imshow(img)

# 我们在特征图（fmap）上⽣成锚框（anchors），每个单位（像素）作为锚框的中⼼。
# 由于锚框中的(x, y)轴坐标值（anchors）已经被除以特征图（fmap）的宽度和⾼度，
# 因此这些值介于0和1之间，表⽰特征图中锚框的相对位置。
def display_anchors(fig, fmap_w, fmap_h, scale):
	my_plt.set_figsize()
	# 前两个维度上的值不影响输出
	fmap = torch.zeros((1, 10, fmap_h, fmap_w))
	anchors = test_anchor.multibox_prior(fmap, sizes=scale, ratios=[1, 2, 0.5])
	bbox_scale = torch.tensor((w, h, w, h))
	test_anchor.show_bboxes(fig.axes,
					anchors[0] * bbox_scale)

# 为了在显⽰时更容易分辨，在这⾥具有不同中⼼的锚框不会重叠：
# 锚框的尺度设置为0.15，特征图的⾼度和宽度设置为4。
# 我们可以看到，图像上4⾏和4列的锚框的中⼼是均匀分布的。
# 也就是每行四组，一共四行，总共十六组。
# 每一个组有三个不同宽⾼⽐的锚框。注意以同⼀像素为中⼼的锚框的数量是len(scale) + len(ratios) − 1。
display_anchors(figPic, fmap_w=4, fmap_h=4, scale=[0.15])
my_plt.plt.show()

# 然后，我们将特征图的⾼度和宽度减⼩⼀半，然后使⽤较⼤的锚框来检测较⼤的⽬标。
# 当尺度设置为0.4时，⼀些锚框将彼此重叠。
figPic = my_plt.plt.imshow(img)
display_anchors(figPic, fmap_w=2, fmap_h=2, scale=[0.4])
my_plt.plt.show()

# 最后，我们进⼀步将特征图的⾼度和宽度减⼩⼀半，然后将锚框的尺度增加到0.8。
# 此时，锚框的中⼼即是图像的中⼼。
figPic = my_plt.plt.imshow(img)
display_anchors(figPic, fmap_w=1, fmap_h=1, scale=[0.8])
my_plt.plt.show()


