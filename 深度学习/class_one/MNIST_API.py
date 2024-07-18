import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import keras.layers

# print(tf.__version__)
# 加载数据集
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()

print(train_image.shape)
print(train_label.shape)

# 展示图片
plt.imshow(train_image[0])
plt.show()
# 图片最大值
np.max(train_image[0])
# 图片最小值
np.min(train_image[0])
# 第一张图片的标签值
print(train_label[0])

# 数据归一化
train_image = train_image / 255.0
test_image = test_image / 255.0

# 输入层
input = keras.Input(shape=(28, 28))
# 第二层拉成28*28的序列
x = keras.layers.Flatten()(input)
# 第二层 隐藏层
x1 = keras.layers.Dense(32, activation='relu')(x)
# 第三层 dropout层
x2 = keras.layers.Dropout(0.1)(x1)
# 第四层 Dense层
x3 = keras.layers.Dense(64, activation='relu')(x2)
# 输出层
output = keras.layers.Dense(10, activation='softmax')(x3)
model = keras.Model(inputs=input, outputs=output)
model.summary()

