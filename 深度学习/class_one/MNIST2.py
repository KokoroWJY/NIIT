import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

tf.__version__

# 加载数据集
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
# train_image的shape
train_image.shape
# train_label的shape
train_label.shape
# 数据归一化
train_image = train_image / 255
test_image = test_image / 255
# train做了one-hot编码
train_label_onehot = tf.keras.utils.to_categorical(train_label)

train_label_onehot[0]
# test做了one-hot编码
test_label_onehot = tf.keras.utils.to_categorical(test_label)

test_label_onehot

# 建立网络结构
model = tf.keras.Sequential()
# 建立输入层，将图片拉成了一维数组作为网络的输入
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# 建立第二层：隐含层（神经元节点个数；激活函数）
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 建立第三层：隐含层
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 建立第四层（输出层，节点个数为10，激活函数用的softmax）
# model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# 网络结构
model.summary()
# 设置网络参数（优化器,lOSS,评价指标）
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['acc']
#               )

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['acc']
              )

# 网络训练
# model.fit(train_image,train_label_onehot,epochs=5)

history = model.fit(train_image, train_label_onehot, epochs=10,
                    validation_data=(test_image, test_label_onehot))
# plot损失值
history.history.keys()

plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
# plot准确率
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
plt.show()