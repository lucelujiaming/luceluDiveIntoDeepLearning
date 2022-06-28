'''
试图解决的问题是对灰度图像进行分类的手写数字(28×28个像素)到他们的10个分类(0到9)。
'''
# 导入数据
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 了解数据情况
#test_images.shape # (10000, 28, 28)
#test_labels # array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)

# 将输入数组形状由(60000,28,28)转换为(60000,28 * 28)
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

# 将[0,255]区间的整数转换为[0,1]之间的浮点数
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 对分类标签y进行one-hot编码
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


from keras import models
from keras import layers
network = models.Sequential()

#我们的网络由两个密集层（Dense layers）组成，它们紧密连接(也称为完全连接)神经层。
#第二个(也是最后一个)层是10路softmax层，这意味着它将返回一个包含10个概率分数的数组(总和为1)，
#每个分数都将是当前数字图像属于我们的10个分类之一的概率。
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

#为了使网络为训练做好准备，我们需要再选择三样东西，作为编译步骤的一部分:
#损失函数： 网络如何能够测量它在训练数据上的表现，如何能够引导自己走向正确的方向
#优化器：网络根据所接收的数据及其损失函数进行自我更新的机制
#监控指标：这里，我们在训练和测试期间只关心准确性(正确分类的图像的一部分)
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#训练这个网络
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#检查模型在测试集上的表现是否良好
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
