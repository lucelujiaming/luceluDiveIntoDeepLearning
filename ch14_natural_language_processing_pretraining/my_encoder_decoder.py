from torch import nn

# 只指定⻓度可变的序列作为编码器的输⼊X。
# 任何继承这个Encoder基类的模型将完成代码实现。
#@save
class Encoder(nn.Module):
	"""编码器-解码器架构的基本编码器接⼝"""
	def __init__(self, **kwargs):
		super(Encoder, self).__init__(**kwargs)
	def forward(self, X, *args):
		raise NotImplementedError
#@save
class Decoder(nn.Module):
	"""编码器-解码器架构的基本解码器接⼝"""
	def __init__(self, **kwargs):
		super(Decoder, self).__init__(**kwargs)
	# 解码器接⼝中，我们新增⼀个init_state函数，
	# ⽤于将编码器的输出（enc_outputs）转换为编码后的状态。
	# 此步骤可能需要额外的输⼊。
	def init_state(self, enc_outputs, *args):
		raise NotImplementedError
	def forward(self, X, state):
		raise NotImplementedError

#@save
class EncoderDecoder(nn.Module):
	"""编码器-解码器架构的基类"""
	# # 包含了⼀个编码器和⼀个解码器，并且还拥有可选的额外的参数。
	def __init__(self, encoder, decoder, **kwargs):
		super(EncoderDecoder, self).__init__(**kwargs)
		self.encoder = encoder
		self.decoder = decoder
	def forward(self, enc_X, dec_X, *args):
		enc_outputs = self.encoder(enc_X, *args)
		# 在前向传播中，编码器的输出⽤于⽣成编码状态，
		dec_state = self.decoder.init_state(enc_outputs, *args)
		# 这个状态⼜被解码器作为其输⼊的⼀部分。
		return self.decoder(dec_X, dec_state)
