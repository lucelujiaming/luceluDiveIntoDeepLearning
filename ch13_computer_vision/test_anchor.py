import torch

import my_plt
import test_bounding_box

if __name__ == '__main__':
	# torch.set_printoptions(2) # 精简输出精度
	torch.set_printoptions(2,sci_mode=False) # 精简输出精度

# 什么是锚框：
#   目标检测算法，顾名思义我们需要在输入图像上检测是否存在我们关注的目标。
#   因此我们需要在输入图像上进行大量的采样，然后进行判断是否存在目标，
#   并调整区域边界从而更准确的预测目标的真实边框。
#   故在图像上的大量采样所得到的不同缩放比和宽高比的边界框就称为锚框。

# 注意：这里的尺寸是做过归一化的。也就是图片的尺寸为1 * 1。
#      因此上这里所有的参数和变量的单位都是百分比。
# 这里仔细说一下缩放比和宽高比。英文写做scales和ratios。
#   scales代表的是anchor box在输入图片的box大小，即面积为scales^2的box。
#   ratios代表在面积恒定的条件下，对宽高进行缩放，使宽高比例为ratios的box。

# 为了⽣成多个不同形状的锚框，让我们设置许多缩放⽐（scale）取值[s1, ..., sn]和
# 许多宽⾼⽐（aspect ratio）取值[r1, ..., rm]。
# 当使⽤这些⽐例和⻓宽⽐的所有组合以每个像素为中⼼时，输⼊图像将总共有whnm个锚框。
# 但是，这会导致计算复杂性很容易过⾼。
# 在实践中，我们只考虑包含s1或r1的组合，该组合由公式(13.4.1)给出：
#      (s1, r1),(s1, r2), ...,(s1, rm),(s2, r1),(s3, r1), ...,(sn, r1). 
# 也就是说，这个公式(13.4.1)可以分解为两个部分。
#   第一个部分是缩放⽐的第一个元素s1和所有宽⾼⽐[r1, ..., rm]的组合。也就是：
#      (s1, r1),(s1, r2), ...,(s1, rm)
#   第二个部分是缩放⽐的其他元素和宽⾼⽐的第一个元素r1的组合。也就是：
#      (s2, r1),(s3, r1), ...,(sn, r1)

# 下面给出几个例子。例如：
#     scales如果为0.15，因为图片的尺寸为1 * 1。表明图片的面积为(0.15 * 0.15) = 0.225。
#     但是我们还设定了ratios。也就是宽⾼比。
#     如果ratios为[1.0000, 2.0000, 0.5000]，因此上，我们就得到了三个组合。
#         [[0.15, 1.0], [0.15, 2.0], [0.15, 0.5]]
#     注意这里给出的ratios是精心设计过的。其中，第一个元素为1，而第二个元素和第三个元素互为倒数关系。
#     因此上，[1.0000, 2.0000, 0.5000]表明，我们需要三种宽⾼比，分别是1:1, 2;1, 1:2。
#     针对这三个组合，我们计算结果如下：
#         [[0.1500, 0.2121, 0.1061], [0.1500, 0.1061, 0.2121]]
#     上面的计算结果满足这样的关系：
#         0.15 * 0.15 = 0.2121 * 0.1061 = 0.1061 * 0.2121 = 0.225
#         0.15   / 0.15   = 1
#         0.2121 / 0.1061 = 2
#         0.1061 / 0.2121 = 0.5
# 为了便于理解，我们再给出一个多缩放比的例子。
#     如果scales为[0.1500, 0.2000]。因为图片的尺寸为1 * 1。
#     表明图片的面积为[(0.15 * 0.15), (0.2 * 0.2)] = [0.225, 0.04]。
#     而ratios仍为[1.0000, 2.0000, 0.5000]。
#     我们会得到四个组合。分别是缩放比的第一个元素s1和所有宽⾼比的组合[(s1, r1), (s1, r2), (s1, r3)]与
#     缩放比的其他元素和宽⾼比的第一个元素r1的组合(s2, r1)。因此上，我们得到下面的结果：
#         [(s1, r1),    (s2, r1),   (s1, r2),    (s1, r3)] 
#         [[0.15, 1.0], [0.2, 1.0], [0.15, 2.0], [0.15, 0.5]]
#     针对这四个组合，我们计算结果如下：
#         [[0.1500, 0.2000, 0.2121, 0.1061], [0.1500, 0.2000, 0.1061, 0.2121]]
#     上面的计算结果满足这样的关系：
#         0.15 * 0.15 = 0.2121 * 0.1061 = 0.1061 * 0.2121 = 0.225
#         0.15   / 0.15   = 1
#         0.2121 / 0.1061 = 2
#         0.1061 / 0.2121 = 0.5
#         0.2  * 0.2  = 0.04
#         0.2  / 0.2  = 1

# 上述⽣成锚框的⽅法。我们指定输⼊图像、尺⼨列表和宽⾼⽐列表，
# 然后此函数将返回所有的锚框。
# 实现思路
#   1）生成中心点坐标位置矩阵
#   2）生成锚框对于每个中心点的宽高矩阵
#   3）两者进行相加，得到锚框矩阵
# 传入的参数如下：
#    data - 这个是一个四元数。前两个元素没有用到。后面两个元素，给出了每行和每列几组锚框
#    sizes, ratios - 尺⼨列表和宽⾼⽐列表。
#@save
def multibox_prior(data, sizes, ratios):
	"""⽣成以每个像素为中⼼具有不同形状的锚框"""
	# 第一步：生成我们需要的辅助变量
	# 这行代码看起来好像是获得图像的高和宽。但是只要跟过代码就知道，
	# 这里得到的是锚框的行数和列数。
	in_height, in_width = data.shape[-2:]
	# 获得尺寸列表长度，宽⾼⽐列表长度。
	# num_sizes, num_ratios 为输入的边长比和宽高比数量。
	device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
	# 每个中⼼点都将有“boxes_per_pixel”个锚框。
	# 以同⼀像素为中⼼的锚框的数量是n + m − 1。
	# boxes_per_pixel 为每个像素点生成的锚框数量
	boxes_per_pixel = (num_sizes + num_ratios - 1)
	# 把传入的尺寸列表转换为张量。
	size_tensor = torch.tensor(sizes, device=device)
	# 把传入的宽⾼⽐列表转换为张量。
	ratio_tensor = torch.tensor(ratios, device=device)
	# 为了将锚点移动到像素的中⼼，需要设置偏移量。
	# 因为⼀个像素的的⾼为1且宽为1，我们选择偏移我们的中⼼0.5
	offset_h, offset_w = 0.5, 0.5
	# 计算锚框排布的步长。
	steps_h = 1.0 / in_height # 在y轴上缩放步⻓
	steps_w = 1.0 / in_width  # 在x轴上缩放步⻓

	# 第二步：生成坐标中心矩阵
	# 如下代码存在一个尺度问题，也就是x匹配y， 还是y匹配x的选择问题，如下代码中选择的是x匹配y。
	# ⽣成锚框的所有中⼼点。
	# center_h, center_w 为缩放后的y轴，x轴的中心点位置。
	# 这里的逻辑很简单。假设我们准备生成4 * 4的锚框。因为图片的尺寸做过归一化，为1 * 1。
	# 因此上，锚框的所有中心点的坐标就是：
	#       ([0, 1, 2, 3] + 0.5) * (1/4)  
	#     = [0.5, 1.5, 2.5, 3.5] * (1/4)
	#     = [0.1250, 0.3750, 0.6250, 0.8750]
	# X方向和Y方向使用上面同样的公式计算。
	center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
	center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
	# print("display_anchors::center_h, center_w : ", center_h, center_w)
	# 用两个坐标轴上的点在平面上画网格。
	# shift_y, shift_x 为 将一维的tensor转换为 h * w 的二维tensor
    # 其中shift_y 的每列为center_y。shift_x 的每行为center_w。
    # 这里使用了meshgrid函数。这个函数原本的作用是绘制二维的坐标网格。
    # 也就是以输入的第一个参数列表的元素作为X坐标，第二个参数列表的元素作为Y坐标的一个网格。
    # 返回的是这个网格上每一个点的坐标。这个说起来有点晦涩。其实只要做一次绘图就明白了。
    # 这里的计算结果如下：
    #     center_h : tensor([0.1250, 0.3750, 0.6250, 0.8750])
    #     center_w : tensor([0.1250, 0.3750, 0.6250, 0.8750])
    # 得到：
    #   shift_y : tensor([[0.1250, 0.1250, 0.1250, 0.1250],
    #                     [0.3750, 0.3750, 0.3750, 0.3750],
    #                     [0.6250, 0.6250, 0.6250, 0.6250],
    #                     [0.8750, 0.8750, 0.8750, 0.8750]])
    #   shift_x : tensor([[0.1250, 0.3750, 0.6250, 0.8750],
    #                     [0.1250, 0.3750, 0.6250, 0.8750],
    #                     [0.1250, 0.3750, 0.6250, 0.8750],
    #                     [0.1250, 0.3750, 0.6250, 0.8750]])
	shift_y, shift_x = torch.meshgrid(center_h, center_w)
	# # 将其拉成一维tensor shape = [in_heifht * in_width]
	shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1) 

	# 第三步：创建锚框宽高矩阵
	# ⽣成“boxes_per_pixel”个⾼和宽，
	# 之后⽤于创建锚框的四⻆坐标(xmin,xmax,ymin,ymax)
	"""生成w， h的一维tensor"""
	# 有了前面给出的例子，这里的计算就比较好理解了。
	#     这里的size_tensor * torch.sqrt(ratio_tensor[0])计算的是缩放比的所有元素和宽高比的第一个元素r1的组合。
	#     计算出来就是(s1, r1), (s2, r1)
	#     而sizes[0] * torch.sqrt(ratio_tensor[1:])计算的是缩放比的第一个元素s1和宽高比的其他元素的组合。
	#     计算出来就是(s1, r2), (s1, r3)
	#     这个计算方法得到的结果和前面例子中依据的公式(13.4.1)是等效的。
	#     另外，因为是宽高比。因此上，计算w的时候，乘以宽高比。而计算h的时候，就需要除以宽高比。
	w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
				sizes[0] * torch.sqrt(ratio_tensor[1:])))\
				* in_height / in_width # 处理矩形输⼊
	h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
				sizes[0] / torch.sqrt(ratio_tensor[1:])))

	# 前面我们已经依据公式(13.4.1)计算出来了缩放比和宽高比的组合。
	# 下面，我们首先计算，当中心点位于(0, 0)的时候，基于这些组合的坐标。计算方法非常简单，就是取正负值以后除以2。
	# 例如，如果scales为0.15且ratios为[1.0000, 2.0000, 0.5000]的时候，我们得到了这样的组合。
	#         [[0.1500, 0.2121, 0.1061], [0.1500, 0.1061, 0.2121]]
	# 这里面包括三个锚框：宽和高分别是[(0.15, 0.15), (0.2121, 0.1061), (0.1061, 0.2121)]
	# 如果当中心点位于(0, 0)的时候，这三个框的坐标就分别是：
	#       [[-0.0750, -0.0750,  0.0750,  0.0750],
	#        [-0.1061, -0.0530,  0.1061,  0.0530],
	#        [-0.0530, -0.1061,  0.0530,  0.1061]]
	# 计算方法就是把三个锚框的宽和高除以2以后，取正负号。这就是下面的代码的含义。
	#       torch.stack((-w, -h, w, h)) / 2 
	# 而因为我们需要建立的锚框是每行in_width个，一共in_height行。
	# 因此上，我们需要重复上面代码返回的结果in_height * in_width次。这就是下面的代码的含义。
	#       repeat(in_height * in_width, 1)
	# 上面两个算式连接起来，把除以2提出来，放在后面。就得到下面的代码了。
	# 除以2来获得半⾼和半宽
	anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
					in_height * in_width, 1) / 2 

	# 将得到的shift_y, shift_x 进行组合得到每个像素点的坐标矩阵
	# 其中的难点为 tensor.repeat() 和 tensor.repeat_interleave()的区别，这里个人理解是
    #     tensor.repeat(*size) ：将tensor看作一个元素，生成*size的尺寸
    #     tensor.repeat_interleave() : 对于输入，在指定维度上进行重复

	# 每个中⼼点都将有“boxes_per_pixel”个锚框，
	# 所以⽣成含所有锚框中⼼的⽹格，重复了“boxes_per_pixel”次
    # 下面是具体的生成含所有锚框中心的网格的步骤。
    # 首先根据前面的计算，我们利用meshgrid函数得到了4行4列的16组坐标。
    # 之后调用reshape拉成一维tensor。分别包含16个元素的X坐标和Y坐标。
    #   shift_x = tensor([0.1250, 0.3750, 0.6250, 0.8750, 
    #                     0.1250, 0.3750, 0.6250, 0.8750,
    #                     0.1250, 0.3750, 0.6250, 0.8750, 
    #                     0.1250, 0.3750, 0.6250, 0.8750])
    #   shift_y = tensor([0.1250, 0.1250, 0.1250, 0.1250, 
    #                     0.3750, 0.3750, 0.3750, 0.3750, 
    #                     0.6250, 0.6250, 0.6250, 0.6250, 
    #                     0.8750, 0.8750, 0.8750, 0.8750])
    # 之后我们调用torch.stack按照[shift_x, shift_y, shift_x, shift_y]进行一维堆叠。
    # 这样堆叠的原因是，我们最后用于绘制边界框的test_bounding_box.bbox_to_rect接受的坐标格式是：
    #        (左上x, 左上y, 右下x, 右下y)
    # 也就是这段代码：
    #     torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
    # 以上面得到的shift_y和shift_x为例。堆叠结果就是：
    #   tensor([shift_x[ 0], shift_y[ 0], shift_x[ 0], shift_y[ 0]], 
    #          [shift_x[ 1], shift_y[ 1], shift_x[ 1], shift_y[ 1]], 
    #          [shift_x[ 2], shift_y[ 2], shift_x[ 2], shift_y[ 2]],
    #          ......                                           
    #          [shift_x[15], shift_y[15], shift_x[15], shift_y[15]])
    # 之后我们调用repeat_interleave在第一个维度上堆叠。堆叠结果就是：
    #   tensor([shift_x[ 0], shift_y[ 0], shift_x[ 0], shift_y[ 0]], 
    #          [shift_x[ 0], shift_y[ 0], shift_x[ 0], shift_y[ 0]], 
    #          [shift_x[ 0], shift_y[ 0], shift_x[ 0], shift_y[ 0]], 
    #          [shift_x[ 1], shift_y[ 1], shift_x[ 1], shift_y[ 1]], 
    #          [shift_x[ 1], shift_y[ 1], shift_x[ 1], shift_y[ 1]], 
    #          [shift_x[ 1], shift_y[ 1], shift_x[ 1], shift_y[ 1]], 
    #          [shift_x[ 2], shift_y[ 2], shift_x[ 2], shift_y[ 2]],
    #          ......                                           
    #          [shift_x[15], shift_y[15], shift_x[15], shift_y[15]])
    # 这样堆叠的原因是和前面anchor_manipulations的堆叠保持一致。
    # 结果就是我们首先在4行4列的16组坐标的第一个元素的位置，绘制三次不同缩放比和宽高比的边界框。
    # 之后移动第二个元素，再次绘制三次。以此类推。
	out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
				dim=1).repeat_interleave(boxes_per_pixel, dim=0)
	# 第四步：相加返回
	# 有了有了所有锚框中心的网格，和当中心点位于(0, 0)的时候，基于缩放比和宽高比的组合的坐标。
	# 我们只需要把这两组数据加起来，就可以得到边界框坐标了。
	output = out_grid + anchor_manipulations
	return output.unsqueeze(0)

if __name__ == '__main__':
	# 我们可以看到返回的锚框变量Y的形状是（批量⼤⼩，锚框的数量，4）。
	img = my_plt.plt.imread('../img/catdog.jpg')
	h, w = img.shape[:2]
	print(h, w)
	# 显示图片大小为700 * 600
	fig = my_plt.plt.imshow(img)
	# my_plt.plt.show()

	X = torch.rand(size=(1, 3, h, w))
	Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
	print("Y.shape : ", Y.shape)

	# 将锚框变量Y的形状更改为(图像⾼度,图像宽度,以同⼀像素为中⼼的锚框的数量,4)后，
	# 我们可以获得以指定像素的位置为中⼼的所有锚框。
	boxes = Y.reshape(h, w, 5, 4)
	# 在接下来的内容中，我们访问以（250,250）为中⼼的第⼀个锚框。
	# 它有四个元素：锚框左上⻆的(x, y)轴坐标和右下⻆的(x, y)轴坐标。
	# 将两个轴的坐标各分别除以图像的宽度和⾼度后，
	# 所得的值介于0和1之间。
	print("boxes[250, 250, 0, :] : ", boxes[250, 250, 0, :])

# 为了显⽰以图像中以某个像素为中⼼的所有锚框，
# 我们定义了下⾯的show_bboxes函数来在图像上绘制多个边界框。
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
	"""显⽰所有边界框"""
	def _make_list(obj, default_values=None):
		if obj is None:
			obj = default_values
		elif not isinstance(obj, (list, tuple)):
			obj = [obj]
		return obj
	# 构建标签列表。
	labels = _make_list(labels)
	# 构建配色方案。如果指定了配色方案。就使用这个配色方案。否则使用默认的配色方案。
	colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
	for i, bbox in enumerate(bboxes):
		# 选定锚框颜色。
		color = colors[i % len(colors)]
		# 选定锚框大小。
		rect = test_bounding_box.bbox_to_rect(bbox.detach().numpy(), color)
		axes.add_patch(rect)
		if labels and len(labels) > i:
			# 设定文字颜色。
			text_color = 'k' if color == 'w' else 'w'
			# 绘制边界框。
			axes.text(rect.xy[0], rect.xy[1], labels[i],
				va='center', ha='center', fontsize=9, color=text_color,
				bbox=dict(facecolor=color, lw=0))

if __name__ == '__main__':
	my_plt.set_figsize()
	bbox_scale = torch.tensor((w, h, w, h))
	fig = my_plt.plt.imshow(img)

	# 现在，我们可以绘制出图像中所有以(250,250)为中⼼的锚框了。
	# 如下所⽰，缩放⽐为0.75且宽⾼⽐为1的蓝⾊锚框很好地围绕着图像中的狗。
	show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
			['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 
			's=0.75, r=2', 's=0.75, r=0.5'])
	# 显示加了锚框的图片。
	my_plt.plt.show()

# 我们将使⽤交并⽐来衡量锚框和真实边界框之间、以及不同锚框之间的相似度。
# 给定两个锚框或边界框的列表，以下box_iou函数将在这两个列表中计算它们成对的交并⽐。
# 传入的两个参数是两个边界框列表。格式为(top, left, bottom, right)
# 第一个是所有锚框的列表。第二个是所有真实边界框的列表。
# 实现思路
#    1）得到锚框和真实边界框的面积
#    2）得到交集的左上和右下顶点坐标
#    3）计算每一个锚框和所有真实边款的交集，并集面积
#    4）将得到的面积与并集相除得到IOU值矩阵
# 虽然这里没有区分锚框和真实边界框。但是在assign_anchor_to_bbox中：
#    boxes1 - 锚框矩形坐标列表。
#    boxes2 - 真实边界框矩形坐标列表。
#@save
def box_iou(boxes1, boxes2):
	# print("box_iou::boxes1 : ", boxes1)
	# print("box_iou::boxes2 : ", boxes2)
	"""计算两个锚框或边界框列表中成对的交并⽐"""
	# 计算框的面积。计算方法就是长乘以宽。
	# 也就是(right - left) * (bottom - top)
	box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
								(boxes[:, 3] - boxes[:, 1]))
	# 1）得到锚框和真实边框的面积
	# boxes1,boxes2,areas1,areas2的形状:
	# boxes1：(boxes1的数量,4),
	# boxes2：(boxes2的数量,4),
	# areas1：(boxes1的数量,),
	# areas2：(boxes2的数量,)
	# 计算得到所有锚框的面积
	areas1 = box_area(boxes1)
	# 计算得到所有真实边界框的面积
	areas2 = box_area(boxes2)

	# 2）得到交集的左上和右下顶点坐标
	'''
        这里运用的广播机制
        boxes1.shape : [anchors_num, 4]
        boxes2.shape : [classes_num, 4]
        boxes1[:, None, :2].shape : [anchors_num, 1, 2]
        torch.max(boxes1[:, None, :2], boxes2[:, :2]).shape 
        = [anchors_num, classes_num, 2]
        通过广播机制能够将每个锚框与所有的真是边框进行计算，
        也就是一个锚框与classes_num种真实边框进行计算。
	'''
	# inter_upperlefts,inter_lowerrights,inters的形状:
	# (boxes1的数量,boxes2的数量,2)
	# 取出boxes1和boxes2前两列，也就是(top, left)。
	# 通过广播机制，把锚框与真实边框进行计算。

	# torch.max(tensor1,tensor2) element-wise 
	# 比较tensor1 和tensor2 中的元素，返回较大的那个值。
	# 这个torch.max的逻辑是这样的。
	# 首先取出boxes1[:, None, :2]的第一个元素[0.00, 0.10]。
	# 和boxes2[:, :2]进行对比。boxes2[:, :2]为[[0.10, 0.08], [0.55, 0.20]]
	# 为了成功和boxes2[:, :2]进行对比，扩展[0.00, 0.10]为[[0.00, 0.10], [0.00, 0.10]]
	# 比较过程如下：
	#     boxes1[:, None, :2]: [[0.00, 0.10], [0.00, 0.10]]
	#           boxes2[:, :2]: [[0.10, 0.08], [0.55, 0.20]]
	#  之后取这四个数的最大值: [[0.10, 0.10], [0.55, 0.20]]
	# 这就得到了第一个元素。后面的五个以此类推。
	# 这样我们就得到了10个矩形坐标，按照传入的boxes1个数，也就是锚框个数分成5组。
	# 每组两个矩形坐标。对应传入的boxes1个数，也就是真实边界框个数。
	inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
	# 取出boxes1和boxes2后两列，也就是(bottom, right)。
	# 这个过程和上面类似。torch.min取最小值。例如第一个元素：
	# 首先取出boxes1[:, None, 2:]的第一个元素[0.20, 0.30]。
	# 和boxes2[:, 2:]进行对比。boxes2[:, 2:]为[[0.52, 0.92], [0.90, 0.88]]
	# 为了成功和boxes2[:, 2:]进行对比，扩展[0.20, 0.30]为[[0.20, 0.30], [0.20, 0.30]]
	# 比较过程如下：
	#     boxes1[:, None, 2:]: [[0.20, 0.30], [0.20, 0.30]]
	#           boxes2[:, 2:]: [[0.52, 0.92], [0.90, 0.88]]
	#  之后取这四个数的最小值: [[0.20, 0.30], [0.20, 0.30]]
	inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

	# 3）计算每一个锚框和所有真实边款的交集，并集面积
	# 其中每一行为一个锚框与每个真实边框交集的宽高。
	# tensor.clamp(min=0)的意思是如果值为负数则设为0，因为得到的交集的宽度和高度 >= 0
	# 相减以后，如果出现负值，就设置为零。
	inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
	# inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
	'''
	    inters.shape : [anchors_num, class_num, 2]
	    提取宽度：inters[:, :, 0].shape : [anchors_num, classes_num]
	    提取高度：inters[:, :, 1].shape : [anchors_num, classes_num]
	    其中每一行为一个锚框与每个真实边框交集的宽高
	'''
	# 计算交集面积。得到10个面积值。
	inter_areas = inters[:, :, 0] * inters[:, :, 1]
	# 计算并集面积
	# 这里依然运用了广播机制，将第一个锚框areas1[:, None]与每种真实边框areas2相加，之后减去交集。
	union_areas = areas1[:, None] + areas2 - inter_areas
	# 4）将得到的交集面积与并集相除得到IOU值矩阵
	return inter_areas / union_areas

# 把最接近的真实边界框分配给锚框。
# 在拥有了对锚框的量化标准后，就可以通过算法来进行锚框的选择和标号。
# 在锚框的标号中，我们采取两步：
#    a. 选出当前IOU矩阵的最大值，将其下标进行存储，然后删除所在的行和列，循环执行。
#    b. 设置IOU阈值，将高于IOU阈值的锚框下标进行存储。
# 实现思路
#   1）用一个一维tensor来保存分配结果，下标表示第几个锚框，值表示列（类别）
#   2）由于找到最大的过程会修改IOU值矩阵，因此先找到大于阈值的锚框进行标号存储
#   3）循环找到全局最大值，进行存储
# 传入的参数为：
#   :param ground_truth:  真实框
#   :param anchors: 所有锚框
#   :param device: 设备
#   :param iou_thread: iou限度。交并比IoU的阈值。
#   :return: 锚框列表 索引为i 值为j
# torchvision中已经有了nms
#    torchvision.ops.nms(boxes, scores, iou_threshold)
#       boxes (Tensor[N, 4])  – bounding boxes坐标. 格式：(x1, y1, x2, y2)
#       scores (Tensor[N])    – bounding boxes得分
#       iou_threshold (float) – IoU过滤阈值
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
	"""将最接近的真实边界框分配给锚框"""
	# 第一步：准备需要的数据
	# num_anchors : 锚框数量 num_get_boxes : 真实边框数量
	# 得到锚框个数，真实边界框个数。
	num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0] 
	# 位于第i⾏和第j列的元素x_ij是锚框i和真实边界框j的IoU
	# 每个锚框与真实框的iou值
	# 这里计算出来的iou矩阵如下：
	#     jaccard :  tensor([[0.05, 0.00], 
	#                        [0.14, 0.00],
    #                        [0.00, 0.57], 
    #                        [0.00, 0.21], 
    #                        [0.00, 0.75]])
    # 这个就是13.4.3中提到的iou矩阵。
	jaccard = box_iou(anchors, ground_truth)
	# print("jaccard : ", jaccard)
	# 对于每个锚框，分配的真实边界框的张量。
	# 生成初始一维tensor用来保存锚框标号 初始值为-1 长度为锚框数量。
	# 这个数组保存对于每个锚框，真实边界框的分配结果。
	anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
								device=device)
	# 第二步：找到所有IOU值大于阈值的锚框进行标号。
	# 代码采取的是找到每行中的最大值（即每个锚框最有可能的类别）
	# 根据阈值，决定是否分配真实边界框
	# 返回一行的最大值和索引
	# torth.max(input, dim=None)：返回的是dim维度下的最大值，和其维度索引。
	# 上面计算出来的iou值，分为五组，每组两个。
	# 这里还是五组。但是每组只留下较大的一个。也就是：
	#   tensor([0.05, 0.14, 0.57, 0.21, 0.75])
	# 同时返回torch.max在取最大值的过程中，选取的最大值的下标。也就是：
	#   tensor([0, 0, 1, 1, 1])
	# 因此上，max_ious保存了五组锚框对应的真实边界框下标值。
	max_ious, indices = torch.max(jaccard, dim=1)

	# nonzero 得到非0元素的下标
    # 得到每一行iou值大于0.5的行索引
    #    这里只有[2, 4]符合。
    # torch.nonzero : 返回非0元素的索引 ，按列布局。
	# 也就是满足阈值的锚框。
	anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
	# 得到iou值大于0.5的行索引对应的类别索引。
	# 也就是满足阈值的锚框对应的真实边界框。
	box_j = indices[max_ious >= 0.5]
	# 保存相对应锚框的类别。
	# 这里因为anc_i等于[2, 4]，box_j等于[1, 1]。
	# 因此上，anchors_bbox_map的第2和4个元素被设置为1。
	# 因为满足阈值，因此上利用阈值完成真实边界框分配。
	anchors_bbox_map[anc_i] = box_j

	# 第三步：循环找取全局最大值，并存储
	# 由于IOU值的范围为[0, 1], 因此我们仅需将选中的最大值信息保存后，
    # 将其所在行列的值修改为-1，就等价于删除
    # 创建用于删除的行掩码和列掩码。其实就是一个都是-1的列表。
    # 长度分别是锚框个数和真实边界框个数。
	col_discard = torch.full((num_anchors,), -1)	# 按照行数生成-1
	row_discard = torch.full((num_gt_boxes,), -1)	# 按照列数生成-1
	# 循环寻找最大值，由于我们仅需要找到每种类别的最大值。
	# 因此仅循环真实边界框的数目。
	# 因为一旦我们找到一个边界框。就会把这个边界框的那一列修改为-1。
	# 这样下一轮循环就不会涉及到这个边界框了。
	for _ in range(num_gt_boxes):
		# 将IOU矩阵 flatte然后得到全局最大值。
		# argmax的意思就是把整个矩阵变成一个一维数组，
		# 之后找到这个数组中最大的元素。之后返回他的下标。
		# 对于IOU矩阵来说，最大的元素就是第9个元素0.75。因此上max_idx等于9。
		max_idx = torch.argmax(jaccard)
		# print("max_idx : ", max_idx)
		# 根据返回的最大元素的下标，计算出来这个元素所在行索引和列索引。
		# 方法是：与真实锚框数取余得到列索引，除真实锚框个数得到行索引
		box_idx = (max_idx % num_gt_boxes).long()
		anc_idx = (max_idx / num_gt_boxes).long()
		# 修改对应得存储信息
		anchors_bbox_map[anc_idx] = box_idx
		# 修改其所在行例得值为-1。
		jaccard[:, box_idx] = col_discard
		jaccard[anc_idx, :] = row_discard
	# 第四步： 返回结果。
	# 这个数组中，如果元素为-1.说明这个锚框没有对应的真实边界框。
	# 否则元素值为该元素下标对应的真实边界框。
	return anchors_bbox_map

# 现在我们可以为每个锚框标记类别和偏移量了。
# 锚框A的偏移量将根据B和A中⼼坐标的相对位置以及这两个框的相对⼤⼩进⾏标记。
# 鉴于数据集内不同的框的位置和⼤⼩不同，
# 我们可以对那些相对位置和⼤⼩应⽤变换，使其获得分布更均匀且易于拟合的偏移量。
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
	"""对锚框偏移量的转换"""
	# 把锚框从两⻆表⽰法转换为中⼼宽度表⽰法。因为(13.4.3)需要中⼼坐标。
	c_anc = test_bounding_box.box_corner_to_center(anchors)
	# 把真实边界框从两⻆表⽰法转换为中⼼宽度表⽰法。因为(13.4.3)需要中⼼坐标。
	c_assigned_bb = test_bounding_box.box_corner_to_center(assigned_bb)
	# 参见(13.4.3)
	# 常量的默认值为 σx = σy = 0.1 ，σw = σh = 0.2。
	# 0.1的倒数为10。而0.2的倒数为5。
	offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
	offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
	offset = torch.cat([offset_xy, offset_wh], axis=1)
	return offset
# 我们使⽤真实边界框（labels参数）来标记锚框的类别和偏移量（anchors参数）。
# 传入两个参数：
#     anchors - 锚框列表。每一个列表元素包含四个数字。表示一个矩形框。
#     labels  - 真实边界框。注意：这个列表中每一个列表元素包含五个数字。
#                               第一个表示类别，后面四个数字表示一个矩形框。
#@save
def multibox_target(anchors, labels):
	"""使⽤真实边界框标记锚框"""
	# 获得真实边界框个数，消除锚框的一维特性。
	batch_size, anchors = labels.shape[0], anchors.squeeze(0)
	# 初始化三个列表。
	# 此函数返回三个列表，(bbox_offset, bbox_mask, class_labels)
	#   第一个维度都是batch维度，
	#   bbox_offset返回的是锚框（anchor）与分配给该锚框的真实gt边界框的偏移量
	#             （如果锚框对应的类别是背景的话，偏移量为0），
	#   bbox_mask返回锚框对应类别是背景（0）还是非背景（1）,
	#            mask第二个维度是对应偏移量的四个坐标值。
	#   class_labels是锚框对应的真实类别
	#。             （0是背景，>0是其他类别（原始类别+1了的））
	batch_offset, batch_mask, batch_class_labels = [], [], []
	# 获得锚框的个数。
	device, num_anchors = anchors.device, anchors.shape[0]
	# 遍历每一个真实边界框。
	print("batch_size : ", batch_size)
	for i in range(batch_size):
		label = labels[i, :, :]
		# print("label : ", label)
		# 将最接近的真实边界框分配给锚框。
		# 这里计算出来的结果是：
		#    tensor([-1,  0,  1, -1,  1])
		anchors_bbox_map = assign_anchor_to_bbox(
					label[:, 1:], anchors, device)
		# print("anchors_bbox_map : ", anchors_bbox_map)
		# 这个anchors_bbox_map数组中，如果元素为-1.说明这个锚框没有对应的真实边界框。
		# 否则元素值为该元素下标对应的真实边界框。
		# 因此上，掩码的计算方法如下：首先把-1设置为零。之后把零和正数设置为一。
		# 也就是背景类别的索引设置为零，然后将新类别的整数索引递增⼀。
		# 结果就是：
		#    tensor([[0.], [1.], [1.], [0.], [1.]])
		# 之后调用repeat(1, 4)，复制4份。
		bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4) 
		# print("bbox_mask : ", bbox_mask)
		# 将类标签和分配的边界框坐标初始化为零
		class_labels = torch.zeros(num_anchors, dtype=torch.long,
						device=device)
		assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
						device=device)
		# 使⽤真实边界框来标记锚框的类别。
		# 如果⼀个锚框没有被分配，我们标记其为背景（值为零）
		# 这里是获得IOU矩阵中，分配到真实边界框的锚框索引。
		# 根据上面计算出来分配结果anchors_bbox_map。
		# 只有索引为[[1], 2], [4]]的锚框成功匹配了真实边界框。
		indices_true = torch.nonzero(anchors_bbox_map >= 0)
		# 将背景类别的索引设置为零，
		# 取得分配到的真实边界框索引。
		bb_idx = anchors_bbox_map[indices_true]
		# print("bb_idx : ", bb_idx)
		# 对于对应类别是非背景的锚框，更新其对应的真实gt边界框的类别。非背景的类别+1，背景类别为0
		# 然后将新类别的整数索引递增⼀。
		# class_labels中，给分配到真实边界框的锚框元素(indices_true)赋值。
		# 值等于真实边界框索引加一。
		# 计算方法就是取分配到真实边界框的锚框索引。
		# 根据前面获得的分配好的真实边界框索引，获取label的第一列，
		# 因为label的第一列表示类别。结果就是：
		#     [0, 1, 2, 0, 2]
		class_labels[indices_true] = label[bb_idx, 0].long() + 1
		# 非背景的anchor对应给分配的gt坐标，背景为[0,0,0,0]
		# assigned_bb中，给分配到真实边界框的锚框元素(indices_true)赋值。
		# 计算方法是：
		#    根据前面获得的分配好的真实边界框索引，获取label的剩余列，
		#    因为label的剩余列是矩形坐标。
		assigned_bb[indices_true] = label[bb_idx, 1:]
		# 偏移量转换
		# 锚框与分配给该锚框的真实gt边界框框进行偏移量计算,
		# bbox_mask为将背景的偏移量给归零（*是元素相乘）
		#（类别为背景的assigned_bb值全为0）
		offset = offset_boxes(anchors, assigned_bb) * bbox_mask
		# print("offset : ", offset)
		# 添加一组偏移量。
		batch_offset.append(offset.reshape(-1))
		# 添加一组背景值。
		batch_mask.append(bbox_mask.reshape(-1))
		# 添加一组锚框分配结果。索引为锚框索引，值为分配好的真实边界框索引加一。
		batch_class_labels.append(class_labels)
		# print("batch_offset : ", batch_offset)
		# print("batch_mask : ", batch_mask)
		# print("batch_class_labels : ", batch_class_labels)
	# 将batch维提取出
	bbox_offset = torch.stack(batch_offset)
	bbox_mask = torch.stack(batch_mask)
	class_labels = torch.stack(batch_class_labels)
	# print("bbox_offset : ", bbox_offset)
	# print("bbox_mask : ", bbox_mask)
	# print("class_labels : ", class_labels)
	return (bbox_offset, bbox_mask, class_labels)

if __name__ == '__main__':
	# 真实边界框列表。
	ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
								 [1, 0.55, 0.2, 0.9, 0.88]])
	# 锚框列表。
	anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
							[0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
							[0.57, 0.3, 0.92, 0.9]])
	# 绘制图片
	fig = my_plt.plt.imshow(img)
	# 绘制所有的真实边界框。包含了一只狗和一只猫的两个真实边界框使用黑色绘制。
	show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
	# 绘制所有的锚框。五个锚框使用
	show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
	my_plt.plt.show()

	labels = multibox_target(anchors.unsqueeze(dim=0),
							ground_truth.unsqueeze(dim=0))
	# 第三个元素包含标记的输⼊锚框的类别。
	print("labels[2] : ", labels[2])
	# tensor([[0, 1, 2, 0, 2]])
	# 这个结果分析如下：
	# 首先调用assign_anchor_to_bbox进行处理。逻辑如下：
	#    首先是根据阈值进行处理。第2和4个元素都符合。因此上他们首先过关。
	#    之后使用argmax发现锚框A4与猫的真实边界框的IoU是最⼤的。因此，A4的类别被标记为猫。
	#    在剩下的配对中，锚框A1和狗的真实边界框有最⼤的IoU。因此，A1的类别被标记为狗。
	# 这样就只有1，2，4被赋值。

	# 返回的第⼆个元素是掩码（mask）变量，形状为（批量⼤⼩，锚框数的四倍）。
	# 掩码变量中的元素与每个锚框的4个偏移量⼀⼀对应。
	# 这个变量主要用于对背景的检测。
	# 通过元素乘法，掩码变量中的零将在计算⽬标函数之前过滤掉负类偏移量。
	print("labels[1] : ", labels[1])
	# 返回的第⼀个元素包含了为每个锚框标记的四个偏移值。
	print("labels[0] : ", labels[0])

# 在预测时，我们先为图像⽣成多个锚框，再为这些锚框⼀⼀预测类别和偏移量。
# ⼀个“预测好的边界框”则根据其中某个带有预测偏移量的锚框⽽⽣成。
# 下⾯我们实现了offset_inverse函数，
#@save
def offset_inverse(anchors, offset_preds):
	"""根据带有预测偏移量的锚框来预测边界框"""
	# 该函数将锚框转换为中心坐标。
	anc = test_bounding_box.box_corner_to_center(anchors)
	# 并应⽤(13.4.3)的逆偏移变换来返回预测的边界框坐标。
	pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
	pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
	pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
	# 该函数将预测的边界框坐标转换为四角坐标。
	predicted_bbox = test_bounding_box.box_center_to_corner(pred_bbox)
	return predicted_bbox

# 当有许多锚框时，可能会输出许多相似的具有明显重叠的预测边界框，都围绕着同⼀⽬标。
# 为了简化输出，我们可以使⽤⾮极⼤值抑制（non-maximum suppression，NMS）合并属于同⼀⽬标的类似的预测边界框。
# 以下是⾮极⼤值抑制的⼯作原理。
#    对于⼀个预测边界框B，⽬标检测模型会计算每个类别的预测概率。
#    假设最⼤的预测概率为p，则该概率所对应的类别B即为预测的类别。
#    具体来说，我们将p称为预测边界框B的置信度（confidence）。
#    在同⼀张图像中，所有预测的⾮背景边界框都按置信度降序排序，以⽣成列表L。

# 然后我们通过以下步骤操作排序列表L：
# 1. 从L中选取置信度最⾼的预测边界框B1作为基准，然后将所有与B1的IoU超过预定阈值ϵ的⾮基准预测边界框从L中移除。
#    这时，L保留了置信度最⾼的预测边界框，去除了与其太过相似的其他预测边界框。
#    简⽽⾔之，那些具有⾮极⼤值置信度的边界框被抑制了。
# 2. 从L中选取置信度第⼆⾼的预测边界框B2作为⼜⼀个基准，然后将所有与B2的IoU⼤于ϵ的⾮基准预测边界框从L中移除。
# 3. 重复上述过程，直到L中的所有预测边界框都曾被⽤作基准。
#    此时，L中任意⼀对预测边界框的IoU都⼩于阈值ϵ；因此，没有⼀对边界框过于相似。
# 4. 输出列表L中的所有预测边界框。
# 有三个参数:
#    boxes  - 预测边界框列表
#    scores - 预测边界框的预测概率，也就是工作原理中提到的那个置信度。
#             该列表已经去除了背景的预测概率。之后选取了每一列的最大值。
#    iou_threshold - IoU过滤阈值
# 这里的关键就是这个过滤阈值。因为首先我们取最大的置信度，之后计算这个置信度对应的边界框和其他边界框的IoU。
# 之后使用这个过滤阈值进行过滤。
#@save
def nms(boxes, scores, iou_threshold):
	"""对预测边界框的置信度进⾏排序"""
	print("nms::boxes : ", boxes)
	print("nms::boxes : ", boxes)
	print("nms::scores : ", scores)
	# ⼯作原理：我们通过以下步骤操作排序列表L：
	# 从L中选取置信度最⾼的预测边界框B1作为基准，

	# 将scores从大到小排序。但是不需要排序结果。
    # 只需要重新排序后的元素在排序之前的序号。
	B = torch.argsort(scores, dim=-1, descending=True)
	print("nms::B : ", B)
	keep = [] # 保留预测边界框的指标
	while B.numel() > 0: 
		# 取出第一个得分元素。
		i = B[0]
		keep.append(i)
		# 只有一个元素的时候。退出循环。因为一个元素没法和其他元素计算IoU。
		if B.numel() == 1: 
			break
		# 取出预测边界框列表中的第0个元素boxes[i, :]。
		# 取出预测边界框列表中的剩余元素。这里因为一共有四个预测边界框。因此上剩余元素有三个。
		# 调用box_iou计算第0个预测边界框和另外的所有预测边界框的IOU值。
		# 结果返回包含三个IOU值的列表。在这个例子中，结果如下：
		#   tensor([0.00, 0.74, 0.55])
		iou = box_iou(boxes[i, :].reshape(-1, 4),
				boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
		print("nms::iou : ", iou)

		# 下面将所有与B1的IoU超过预定阈值ϵ的⾮基准预测边界框从L中移除。
		# 这时，L保留了置信度最⾼的预测边界框，去除了与其太过相似的其他预测边界框。
        # 简⽽⾔之，那些具有⾮极⼤值置信度的边界框被抑制了。
        # 重复上述过程，直到只剩下一个元素，而无法计算IoU而退出。

		# 使用IoU过滤阈值过滤IoU计算结果。返回符合条件的IoU列表下标。
		# 在这个例子中，因为iou_threshold为0.5。因此上，计算结果如下：
		#    tensor([0])
		# 注意，这里计算出来的其实还是一个列表。只不过恰好计算出来的结果是一个而已。
		# 例如，如果iou_threshold设定为0.3。计算出来的结果就会变成：
		#    tensor([0, 2])
		# 也就是第0个和第2个元素都满足阈值。
		inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
		# 使用前面计算出来的过滤结果更新order。
		# 因为我们之前计算出来的IoU是预测边界框列表的第0个元素和其他元素的计算结果。
		# 因此上，这里的第0个和第2个元素，其实对应于预测边界框列表中的第1个和第二个元素。
		# 这就是为什么需要加一的原因。例如：
		#   当iou_threshold为0.5时，inds = [0] → inds + 1 = [1] →  B[1] = [3]。
		#   当iou_threshold为0.3时，inds = [0, 2] → inds + 1 = [1, 3] →   B[1, 3] = [3, 2]。
		# 而B = tensor([0, 3, 1, 2])。因此上：
		#   当iou_threshold为0.5时，B[1] = [3]。B就从[0, 3, 1, 2] → [3]
		#   当iou_threshold为0.3时，B[1, 3] = [3, 2]。B就从[0, 3, 1, 2] → [3, 2]
		B = B[inds + 1]
	return torch.tensor(keep, device=boxes.device)

# 我们定义以下multibox_detection函数来将⾮极⼤值抑制应⽤于预测边界框。
# 传入五个参数：
#    cls_probs     - 对于背景、狗和猫其中的每个类的预测概率。
#    offset_preds  - 预测的偏移量。
#    anchors       - 锚框列表。
#    nms_threshold - 用于NMS算法的IOU阈值。
#    pos_threshold - 非背景预测阈值。
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
						pos_threshold=0.009999999):
	"""使⽤⾮极⼤值抑制来预测边界框"""
	print("cls_probs : ", cls_probs)
	print("offset_preds : ", offset_preds)
	print("anchors : ", anchors)
	# 得到预测概率的批次数量。
	device, batch_size = cls_probs.device, cls_probs.shape[0]
	# 做一次降维。
	anchors = anchors.squeeze(0)
	print("anchors : ", anchors)
	# 得到类个数。例如这里包含三个类。分别是背景、狗和猫。
	# 得到锚框列表个数。
	num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
	print("num_classes : ", num_classes)
	print("num_anchors : ", num_anchors)
	out = []
	# 循环批次处理。当然这里只有一批。
	for i in range(batch_size):
		# 取出来第i批预测概率和预测的偏移量。sdfsdf
		cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
		# 找到预测概率中除了背景以外每一列的最大值，和对应的类别索引。
		# 例如，对于预测概率，去除了第一例。结果如下：
		#     tensor([[0.90, 0.80, 0.70, 0.10],    # 狗
        #             [0.10, 0.20, 0.30, 0.90]])   # 猫
        # 每一列的最大值如下：
        #     tensor([0.90, 0.80, 0.70, 0.90])
        # 对应的类别索引： tensor([0, 0, 0, 1])
		conf, class_id = torch.max(cls_prob[1:], 0)
		print("class_id : ", class_id)
		# 利用(13.4.3)的逆偏移变换来返回预测的边界框坐标。
		predicted_bb = offset_inverse(anchors, offset_pred)
		print("predicted_bb : ", predicted_bb)
		# 以使⽤⾮极⼤值抑制（NMS）合并属于同⼀⽬标的类似的预测边界框。
		# 返回我们合并好的预测边界框。
		# 这里是： tensor([0, 3])
		keep = nms(predicted_bb, conf, nms_threshold)
		print("After nms return keep : ", keep)
		print("---------------------------------------------------------- ")
		# 找到所有的non_keep索引，并将类设置为背景。为了实现这个部分。
		# 首先得到所有的边界框下标。也就是tensor([0, 1, 2, 3])
		all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
		# 合并两个张量。得到：tensor([0, 3, 0, 1, 2, 3])
		combined = torch.cat((keep, all_idx))
		# 对于合并后的张量进行去重。同时得到重复个数。去重结果如下：
		#    tensor([0, 1, 2, 3])
		# 结果必然是预测边界框重复次数为2，而剩下的重复次数为1。
		# 重复次数结果如下：
		#    tensor([2, 1, 1, 2])
		uniques, counts = combined.unique(return_counts=True)
		# 得到非预测边界框。也就是预测边界框的补集。结果如下：
		#    tensor([1, 2])
		non_keep = uniques[counts == 1]
		# 合并预测边界框列表和非预测边界框列表。结果如下：
		#    tensor([0, 3, 1, 2])
		# 可以看到，前两个为预测边界框，后两个为非预测边界框。
		all_id_sorted = torch.cat((keep, non_keep))
		# 把非预测边界框的对应类别设置为-1。
		# 也就是把[1, 2]设置为-1。结果如下：
		#     tensor([ 0, -1, -1,  1])
		class_id[non_keep] = -1
		# 之后class_id里面的元素按照all_id_sorted给出的索引重新排布。结果如下：
		#     tensor([ 0,  1, -1, -1])
		class_id = class_id[all_id_sorted]
		# 之后按照同样的方式，把conf里面的元素按照all_id_sorted给出的索引重新排布。结果如下：
		#     tensor([0.90, 0.90, 0.80, 0.70])
		# 把conf里面的元素按照all_id_sorted给出的索引重新排布。结果如下：
		conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
		# pos_threshold是⼀个⽤于⾮背景预测的阈值
		# 小于这个阈值的就是背景。这里都不是的。
		#    tensor([False, False, False, False])
		below_min_idx = (conf < pos_threshold)
		# 小于这个阈值的类别设置为负一。这里因为没有。因此上class_id无变化。
		class_id[below_min_idx] = -1
		print("conf : ", conf)
		conf[below_min_idx] = 1 - conf[below_min_idx]
		print("conf[below_min_idx] : ", conf[below_min_idx])
		print("conf : ", conf)
		# 拼装六元数结果。
		# 最内层维度中的六个元素提供了同⼀预测边界框的输出信息。
		#   第⼀个元素是预测的类索引，从0开始（0代表狗，1代表猫），
		#             值-1表⽰背景或在⾮极⼤值抑制中被移除了。
		#   第⼆个元素是预测的边界框的置信度。
		#   其余四个元素分别是预测边界框左上⻆和右下⻆的(x, y)轴坐标（范围介于0和1之间）。
		pred_info = torch.cat((class_id.unsqueeze(1),
							conf.unsqueeze(1),
							predicted_bb), dim=1)
		print("pred_info : ", pred_info)
		out.append(pred_info)
		print("out : ", out)
	return torch.stack(out)

if __name__ == '__main__':
	# 现在让我们将上述算法应⽤到⼀个带有四个锚框的具体⽰例中。
	anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
							[0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
	# 为简单起⻅，我们假设预测的偏移量都是零，这意味着预测的边界框即是锚框。
	offset_preds = torch.tensor([0] * anchors.numel())
	# 对于背景、狗和猫其中的每个类，我们还定义了四个锚框的预测概率。
	cls_probs = torch.tensor([[0] * 4, # 背景的预测概率
				[0.9, 0.8, 0.7, 0.1], # 狗的预测概率
				[0.1, 0.2, 0.3, 0.9]]) # 猫的预测概率

	# 我们可以在图像上绘制这些预测边界框和置信度。
	fig = my_plt.plt.imshow(img)
	show_bboxes(fig.axes, anchors * bbox_scale,
			['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
	my_plt.plt.show()

	# 现在我们可以调⽤multibox_detection函数来执⾏⾮极⼤值抑制，其中阈值设置为0.5。
	output = multibox_detection(cls_probs.unsqueeze(dim=0),
					offset_preds.unsqueeze(dim=0),
					anchors.unsqueeze(dim=0),
					nms_threshold=0.5)
	# 返回结果的形状是（批量⼤⼩，锚框的数量，6）。
	# 最内层维度中的六个元素提供了同⼀预测边界框的输出信息。
	#   第⼀个元素是预测的类索引，从0开始（0代表狗，1代表猫），
	#             值-1表⽰背景或在⾮极⼤值抑制中被移除了。
	#   第⼆个元素是预测的边界框的置信度。
	#   其余四个元素分别是预测边界框左上⻆和右下⻆的(x, y)轴坐标（范围介于0和1之间）。
	#   可以看出来，anchors的第0个和第3个元素被选中。而第1个和第2个元素被识别为背景。
	#   tensor([[[ 0.00,  0.90,  0.10,  0.08,  0.52,  0.92],
	#            [ 1.00,  0.90,  0.55,  0.20,  0.90,  0.88],
	#            [-1.00,  0.80,  0.08,  0.20,  0.56,  0.95],
	#            [-1.00,  0.70,  0.15,  0.30,  0.62,  0.91]]])
	print("output : ", output)

	# 删除-1类别（背景）的预测边界框后，我们可以输出由⾮极⼤值抑制保存的最终预测边界框。
	fig = my_plt.plt.imshow(img)
	for i in output[0].detach().numpy():
		if i[0] == -1:
			continue
		# 首先根据i[0]打印类别，0表示dog，1表示cat，之后根据i[1]打印置信度。
		label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
		# 之后根据i[2:]打印坐标。因为坐标范围介于0和1之间，因此上需要呈上缩放比例因子。
		show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
	my_plt.plt.show()
