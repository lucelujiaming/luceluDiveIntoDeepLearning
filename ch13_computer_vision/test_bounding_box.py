import torch

import my_plt

if __name__ == '__main__':
	# 下⾯加载本节将使⽤的⽰例图像。
	# 可以看到图像左边是⼀只狗，右边是⼀只猫。它们是这张图像⾥的两个主要⽬标。
	my_plt.set_figsize()
	img = my_plt.plt.imread('../img/catdog.jpg')
	my_plt.plt.imshow(img);
	my_plt.plt.show()

# 边界框是矩形的，由矩形左上⻆的以及右下⻆的x和y坐标决定。
# 另⼀种常⽤的边界框表⽰⽅法是边界框中⼼的(x, y)轴坐标以及框的宽度和⾼度。
# 在这⾥，我们定义在这两种表⽰法之间进⾏转换的函数：
# 从两⻆表⽰法转换为中⼼宽度表⽰法
#@save
def box_corner_to_center(boxes):
	"""从（左上，右下）转换到（中间，宽度，⾼度）"""
	x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
	cx = (x1 + x2) / 2
	cy = (y1 + y2) / 2 
	w = x2 - x1
	h = y2 - y1
	boxes = torch.stack((cx, cy, w, h), axis=-1)
	return boxes
# 从两⻆表⽰法转换为中⼼宽度表⽰法
#@save
def box_center_to_corner(boxes):
	"""从（中间，宽度，⾼度）转换到（左上，右下）"""
	cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
	x1 = cx - 0.5 * w
	y1 = cy - 0.5 * h
	x2 = cx + 0.5 * w
	y2 = cy + 0.5 * h
	boxes = torch.stack((x1, y1, x2, y2), axis=-1)
	return boxes

if __name__ == '__main__':
	# 我们将根据坐标信息定义图像中狗和猫的边界框。
	# 图像中坐标的原点是图像的左上⻆，向右的⽅向为x轴的正⽅向，向下的⽅向为y轴的正⽅向。
	# bbox是边界框的英⽂缩写
	dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], \
						 [400.0, 112.0, 655.0, 493.0]

	# 我们将根据坐标信息定义图像中狗和猫的边界框。
	# bbox是边界框的英⽂缩写
	dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], \
						 [400.0, 112.0, 655.0, 493.0]
	# 我们可以通过转换两次来验证边界框转换函数的正确性。
	boxes = torch.tensor((dog_bbox, cat_bbox))
	print("转换两次 : ", box_center_to_corner(box_corner_to_center(boxes)) == boxes)


# 我们可以将边界框在图中画出，以检查其是否准确。
# 将边界框表⽰成matplotlib的边界框格式。
#@save
def bbox_to_rect(bbox, color):
	# 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
	# ((左上x,左上y),宽,⾼)
	return my_plt.plt.Rectangle(
		xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
		fill=False, edgecolor=color, linewidth=2)

if __name__ == '__main__':
	# 在图像上添加边界框之后，我们可以看到两个物体的主要轮廓基本上在两个框内。
	fig = my_plt.plt.imshow(img)
	fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
	fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
	my_plt.plt.show()

