import torch 

# Non-Maximum Suppression（NMS）非极大值抑制。
# 从字面意思理解，抑制那些非极大值的元素，保留极大值元素。
# 其主要用于目标检测，目标跟踪，3D重建，数据挖掘等。
# 包含三个参数：
#   boxes (Tensor[N, 4])  – 边界框坐标. 格式：(x1, y1, x2, y2)
#   scores (Tensor[N])    – 边界框得分
#   iou_threshold (float) – IoU过滤阈值
def NMS(boxes,scores, thresholds):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # 计算面积
    #   (x2-x1) : tensor([5.0000, 5.0000, 1.6000, 7.9000])
    #   (y2-y1) : tensor([1.9000, 0.8000, 3.0000, 1.0000])
    #   areas   : tensor([9.5000, 4.0000, 4.8000, 7.9000])
    areas = (x2-x1)*(y2-y1)
    # 把输入的scores按照从大到小排序。
    #   sortScores :  tensor([0.5000, 0.4000, 0.3000, 0.2000])
    # 但是不需要排序结果。
    # 只需要重新排序后的元素在排序之前的序号。
    #   order :  tensor([0, 3, 1, 2])
    sortScores, order = scores.sort(0,descending=True)
    keep = []
    # 循环处理边界框得分
    while order.numel() > 0:
        # 只有一个元素的时候。退出循环。
        if order.numel() == 1:
            break
        # 取出第一个得分元素。
        i = order[0]
        keep.append(i)
        # 取出order中除了第一个元素以外的元素，也就是order[1:]。
        # 找到这些元素对应的坐标(x1/y1/x2//y2)[order[1:]]。
        # 如果这些坐标中，有坐标值比order中第一个元素对应的边界框坐标大或者小。
        # 就把这样的元素设置为order中第一个元素对应的坐标(x1/y1/x2//y2)[order[0]]。
        # 例如：
        #   x1 :  tensor([2.0000, 3.0000, 4.0000, 0.1000])
        #   order[1:] :  tensor([3, 1, 2])
        #   因此上，使用order[1:]为坐标，取出x1的值：
        #     x1[order[1:]] :  tensor([0.1000, 3.0000, 4.0000])
        #   接着调用clamp。其设定的最小值为x1[0] :  tensor(2.)
        #   clamp把比x1[i]小的值设定为x1[i]，因此上：
        #     xx1 :  tensor([2., 3., 4.])
        xx1 = x1[order[1:]].clamp(min=x1[i])
        # 同理有：
        #   y1[order[1:]] :  tensor([0., 4., 4.])
        # clamp设定的最小值为y1[0] :  tensor(3.1)。于是：
        #   yy1 :  tensor([3.1000, 4.0000, 4.0000])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        # 同理有：
        #   x2[order[1:]] :  tensor([8.0000, 8.0000, 5.6000])
        # clamp设定的最大值为x2[0] :  tensor(8)。于是：
        #   xx2 :  tensor([7.0000, 7.0000, 5.6000])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        # 同理有：
        #   y2[order[1:]] :  tensor([1.0000, 4.8000, 7.0000])
        # clamp设定的最大值为y2[0] :  tensor(5)。于是：
        #   yy2 :  tensor([1.0000, 4.8000, 5.0000])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        # 计算上面计算出来的，经过了clamp的边界框的面积。
        # 结果如下：
        #    ﻿inter :  tensor([0.0000, 3.2000, 1.6000])
        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h
        # 计算每一个边界框的交并比。

        # 首先我们知道。
        #   areas   : tensor([9.5000, 4.0000, 4.8000, 7.9000])
        # areas[i]为用来做clamp的边界框的面积。这里为tensor(9.5000)
        # areas[order[1:]]为剩下的边界框的面积列表。
        #   tensor([7.9000, 4.0000, 4.8000])
        # 这里使用了广播机制。把areas[i]加到areas[order[1:]]列表上。
        # 因此得到：
        #   tensor([17.4000, 13.5000, 14.3000])
        # 上面我们已经计算出来了经过了clamp的边界框的面积inter。
        # 因此上，差值如下：
        #   tensor([17.4000, 10.3000, 12.7000])
        # 之后我们就可以计算出来inter和这个差值的商。如下：
        #   tensor([0.0000, 0.3107, 0.1260])
        ovr = inter/(areas[i] + areas[order[1:]] - inter)
        # 针对这个计算结果，我们使用IoU过滤阈值进行过滤。
        # 这里的IoU过滤阈值为0.3。可以看出来，只有[0]和[2]满足要求。
        # 因此上：
        #   ids :  tensor([0, 2])
        ids = (ovr<=thresholds).nonzero().squeeze()
        if ids.numel() == 0:
            break
        # 使用起码计算出来的过滤结果更新order。
        # 原来order包含四个元素，内容如下：
        #   order :  tensor([0, 3, 1, 2])
        # 现在只剩满足要求的[0 + 1]和[2 + 1]了。内容如下：
        #   order :  tensor([3, 2])
        order = order[ids+1]
    print("keep : ", keep)
    return torch.LongTensor(keep)

box =  torch.tensor([[  2, 3.1,  7,   5],
                     [  3,   4,  8, 4.8],
                     [  4,   4,5.6,   7],
                     [0.1,   0,  8,   1]]) 
score = torch.tensor([0.5, 0.3, 0.2, 0.4])

keepRet = NMS(box, score, 0.3)


import torch
import torchvision
 
# box =  torch.tensor([[2,3.1,7,5],[3,4,8,4.8],[4,4,5.6,7],[0.1,0,8,1]]) 
# score = torch.tensor([0.5, 0.3, 0.2, 0.4])
 
output = torchvision.ops.nms(boxes=box, scores=score, iou_threshold=0.3)
print('IOU of bboxes:')
iou = torchvision.ops.box_iou(box,box)
print(iou)
print(output)

