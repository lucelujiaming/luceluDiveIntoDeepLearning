import hashlib
import os
import tarfile
import zipfile
import requests

# 下面的下载功能函数来自于kaggle_house_price.py。。
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 下载数据集，将数据集缓存在本地⽬录（默认情况下为../data）中，并返回下载⽂件的名称。
def download(name, cache_dir=os.path.join('..', 'data')): #@save
	"""下载⼀个DATA_HUB中的⽂件，返回本地⽂件名"""
	assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
	# 得到下载地址和文件的sha-1值。
	url, sha1_hash = DATA_HUB[name]
	os.makedirs(cache_dir, exist_ok=True)
	fname = os.path.join(cache_dir, url.split('/')[-1])
	# 如果缓存⽬录中已经存在此数据集⽂件，
	if os.path.exists(fname):
		sha1 = hashlib.sha1()
		with open(fname, 'rb') as f:
			while True:
				data = f.read(1048576)
				if not data:
					break
				sha1.update(data)
				# 并且其sha-1与存储在DATA_HUB中的相匹配，
				# 我们将使⽤缓存的⽂件，以避免重复的下载。
				if sha1.hexdigest() == sha1_hash:
					return fname # 命中缓存
	# 如果文件不存在，就下载。
	print(f'正在从{url}下载{fname}...')
	r = requests.get(url, stream=True, verify=True)
	with open(fname, 'wb') as f:
		f.write(r.content)
	return fname

# 下载并解压
def download_extract(name, folder=None): #@save
	"""下载并解压zip/tar⽂件"""
	fname = download(name)
	base_dir = os.path.dirname(fname)
	# 得到扩展名。
	data_dir, ext = os.path.splitext(fname)
	# 如果是zip文件。调用zipfile解压。
	if ext == '.zip':
		fp = zipfile.ZipFile(fname, 'r')
	# 如果是gz文件。调用tarfile解压。
	elif ext in ('.tar', '.gz'):
		fp = tarfile.open(fname, 'r')
	else:
		assert False, '只有zip/tar⽂件可以被解压缩'
	# 执行解压。
	fp.extractall(base_dir)
	return os.path.join(base_dir, folder) if folder else data_dir
# 下载DATA_HUB中的多个文件。
def download_all(): #@save
	"""下载DATA_HUB中的所有⽂件"""
	for name in DATA_HUB:
		download(name)
