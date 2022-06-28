import torch
import test_torchscript
import train_framework

# 请注意，我们⾄少需要两个GPU来运⾏本节中的实验。
# 下⾯的run函数将执⾏10 次“矩阵－矩阵”乘法时
# 需要使⽤的数据分配到两个变量（x_gpu1和x_gpu2）中，
# 这两个变量分别位于我们选择的不同设备上。
devices = train_framework.try_all_gpus()
print("devices", devices)
def run(x):
	return [x.mm(x) for _ in range(50)]
x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
# IndexError: list index out of range
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])

run(x_gpu1)
run(x_gpu2) # 预热设备
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])
with test_torchscript.Benchmark('GPU1 time'):
	run(x_gpu1)
	torch.cuda.synchronize(devices[0])
with test_torchscript.Benchmark('GPU2 time'):
 	run(x_gpu2)
	torch.cuda.synchronize(devices[1])

# 如果我们删除两个任务之间的synchronize语句，
# 系统就可以在两个设备上⾃动实现并⾏计算。
with test_torchscript.Benchmark('GPU1 & GPU2'):
	run(x_gpu1)
	run(x_gpu2)
	# 在上述情况下，总执⾏时间⼩于两个部分执⾏时间的总和。
	torch.cuda.synchronize()

# 在许多情况下，我们需要在不同的设备之间移动数据，
# ⽐如在CPU和GPU之间，或者在不同的GPU之间。
def copy_to_cpu(x, non_blocking=False):
	return [y.to('cpu', non_blocking=non_blocking) for y in x]
with test_torchscript.Benchmark('在GPU1上运⾏'):
	y = run(x_gpu1)
	# 将会等待⼀个CUDA设备上的所有流中的所有核⼼的计算完成。
	# 函数接受⼀个device参数，代表是哪个设备需要同步。
	# 如果device参数是None（默认值），它将使⽤current_device()找出的当前设备。
	torch.cuda.synchronize()
with test_torchscript.Benchmark('复制到CPU'):
	y_cpu = copy_to_cpu(y)
	# 将会等待⼀个CUDA设备上的所有流中的所有核⼼的计算完成。
	# 函数接受⼀个device参数，代表是哪个设备需要同步。
	# 如果device参数是None（默认值），它将使⽤current_device()找出的当前设备。
	torch.cuda.synchronize()

with test_torchscript.Benchmark('在GPU1上运⾏并复制到CPU'):
	y = run(x_gpu1)
	# 在PyTorch中，to()和copy_()等函数都允许显式的non_blocking参数，
	# 这允许在不需要同步时调⽤⽅可以绕过同步。
	# 设置non_blocking=True让我们模拟这个场景。
	y_cpu = copy_to_cpu(y, True)
	# 将会等待⼀个CUDA设备上的所有流中的所有核⼼的计算完成。
	# 函数接受⼀个device参数，代表是哪个设备需要同步。
	# 如果device参数是None（默认值），它将使⽤current_device()找出的当前设备。
	torch.cuda.synchronize()


