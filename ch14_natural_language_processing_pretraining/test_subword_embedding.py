import collections

# 应⽤⼀种称为字节对编码（Byte Pair Encoding，BPE）的压缩算法来提取⼦词
symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 
           'h', 'i', 'j', 'k', 'l', 'm', 'n', 
           'o', 'p', 'q',      'r', 's', 't', 
           'u', 'v', 'w',      'x', 'y', 'z', 
           '_', '[UNK]']

# 我们只需要⼀个字典raw_token_freqs将词映射到数据集中的频率（出现次数）。
# 注意，特殊符号'_'被附加到每个词的尾部。
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
	token_freqs[' '.join(list(token))] = raw_token_freqs[token]
print("token_freqs : ", token_freqs)

# 我们定义以下get_max_freq_pair函数，其返回词内最频繁的连续符号对，
def get_max_freq_pair(token_freqs):
	pairs = collections.defaultdict(int)
	# 词来⾃输⼊词典token_freqs的键。
	for token, freq in token_freqs.items():
		symbols = token.split()
		for i in range(len(symbols) - 1):
			# “pairs”的键是两个连续符号的元组
			pairs[symbols[i], symbols[i + 1]] += freq
	# 返回词内最频繁的连续符号对
	return max(pairs, key=pairs.get) # 具有最⼤值的“pairs”键
# 作为基于连续符号频率的贪⼼⽅法，
# 字节对编码将使⽤以下merge_symbols函数来合并最频繁的连续符号对以产⽣新符号。
def merge_symbols(max_freq_pair, token_freqs, symbols):
	symbols.append(''.join(max_freq_pair))
	new_token_freqs = dict()
	for token, freq in token_freqs.items():
		# 合并最频繁的连续符号对以产⽣新符号。
		new_token = token.replace(' '.join(max_freq_pair),
								   ''.join(max_freq_pair))
		new_token_freqs[new_token] = token_freqs[token]
	return new_token_freqs

# 现在，我们对词典token_freqs的键迭代地执⾏字节对编码算法。
# 在第⼀次迭代中，最频繁的连续符号对是't'和'a'，因此字节对编码将它们合并以产⽣新符号'ta'。
# 在第⼆次迭代中，字节对编码继续合并'ta'和'l'以产⽣另⼀个新符号'tal'。
num_merges = 10
for i in range(num_merges):
	max_freq_pair = get_max_freq_pair(token_freqs)
	token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
	print(f'合并# {i+1}:',max_freq_pair)

# 在字节对编码的10次迭代之后，我们可以看到列表symbols现在⼜包含10个从其他符号迭代合并⽽来的符号。
print("包含10个从其他符号迭代合并⽽来的符号 : ", symbols)

# 对于在词典raw_token_freqs的键中指定的同⼀数据集，作为字节对编码算法的结果，
# 数据集中的每个词现在被⼦词“fast_”、“fast”、“er_”、“tall_”和“tall”分割。
# 单词“fast er_”和“tall er_”分别被分割为“fast er_”和“tall er_”。
print(list(token_freqs.keys()))

# 作为⼀种贪⼼⽅法，下⾯的segment_BPE函数尝试将单词从输⼊参数symbols分成可能最⻓的⼦词。
def segment_BPE(tokens, symbols):
	outputs = []
	for token in tokens:
		start, end = 0, len(token)
		cur_output = []
		# 具有符号中可能最⻓⼦字的词元段
		while start < len(token) and start < end:
			if token[start: end] in symbols:
				cur_output.append(token[start: end])
				start = end
				end = len(token)
			else:
				end -= 1
		if start < len(token):
			cur_output.append('[UNK]')
		outputs.append(' '.join(cur_output))
	return outputs
# 我们使⽤列表symbols中的⼦词（从前⾯提到的数据集学习）来表⽰另⼀个数据集的tokens。
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))





















