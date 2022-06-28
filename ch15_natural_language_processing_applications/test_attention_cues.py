import torch
import my_plt

# 为了可视化注意⼒权重，我们定义了show_heatmaps函数。
# 其输⼊matrices的形状是（要显⽰的⾏数，要显⽰的列数，查询的数⽬，键的数⽬）。
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
					cmap='Reds', usingION = True):
	"""显⽰矩阵热图"""
	if(usingION == True):
		my_plt.plt.ion()
	my_plt.use_svg_display()
	num_rows, num_cols = matrices.shape[0], matrices.shape[1]
	fig, axes = my_plt.plt.subplots(num_rows, num_cols, figsize=figsize,
				sharex=True, sharey=True, squeeze=False)
	for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
		for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
			pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
			if i == num_rows - 1:
				ax.set_xlabel(xlabel)
			if j == 0:
				ax.set_ylabel(ylabel)
			if titles:
				ax.set_title(titles[j])
	fig.colorbar(pcm, ax=axes, shrink=0.6);
	if(usingION == True):
		my_plt.plt.ioff()
		my_plt.plt.show()
		
if __name__ == '__main__':
	# 在本例⼦中，仅当查询和键相同时，注意⼒权重为1，否则为0。
	attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
	show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries', usingION = False)
	my_plt.plt.ioff()
	my_plt.plt.show()

