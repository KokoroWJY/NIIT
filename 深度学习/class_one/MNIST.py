import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# improt tensorflow.python.keras
from tensorflow import keras

# tf.__version__
# 加载数据集
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()

print(train_image.shape)
print(train_label.shape)

# train的one-hot编码
train_label_onehot = tf.keras.utils.to_categorical(train_label)
print(train_label_onehot[0])
# test的one-hot编码
test_label_onehot = tf.keras.utils.to_categorical(test_label)
print(test_label_onehot)

# 展示图片
plt.imshow(train_image[0])
plt.show()

# 图片最大值
np.max(train_image[0])
# 图片最小值
np.min(train_image[0])
print(train_label[0])
# 数据归一化
train_image = train_image / 255
test_image = test_image / 255
# 建立网络结构
model = tf.keras.Sequential()
# 建立输入层，将图片拉成了一维数组作为网络的输入
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# 建立第二层：隐含层，神经元节点个数
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 建立第三层：隐含层
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 建立第四层（输出层）
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# 设置网络参数（优化器，loss，评价指标）
# model.compile(optimizer = 'adam',#batch_size = 100,
#           loss = 'categorical_crossentropy',
#           metrics=['acc']
#          )

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['acc']
              )

# 开始训练
# model.fit(train_image,train_label_onehot,epochs=5)

history = model.fit(train_image, train_label_onehot, epochs=10,
                    validation_data=(test_image, test_label_onehot))
# plot损失值
history.history.keys()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()
# plot准确度
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
plt.show()

# 网络测试
predict = model.predict(test_image)
# 网络测试结果的shape
print(predict.shape)

print(predict[0])

print(np.argmax(predict[0]))

# model.compile(optimizer = tf.optimizers.Adam(learning_rate=0.001),
#              loss = 'categorical_crossentropy',
#              metrics=['acc'])
