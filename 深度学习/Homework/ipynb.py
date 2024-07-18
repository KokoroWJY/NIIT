# MNIST 手写字符数据集 训练集为 60,000 张 28x28 像素灰度图像，测试集为 10,000 同规格图像，总共 10 类数字标签。
''' 用法：
** ```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() ```
**返回：** 2 个元组： -
**x_train, x_test**: uint8 数组表示的灰度图像，尺寸为 (num_samples, 28, 28)。
**y_train, y_test**: uint8 数组表示的数字标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)。
### 作业要求：
    1. 参考课堂demo，**新建一个ipynb文件。**
    2. 请完成数据加载和数据归一化。
    3. 采用one-hot编码就行标签的预处理
    4. 使用keras构建全连接网络进行分类。（5层的网络），调用API函数
    5. 绘制训练集Loss、ac
'''

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import matplotlib.pyplot as plt

# 数据加载 数据归一化
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 采用one-hot编码就行标签的预处理
y_train_onehot = tf.keras.utils.to_categorical(y_train)
y_test_onehot = tf.keras.utils.to_categorical(y_test)

# 构建网络
# 输入层
input = keras.Input(shape =(28, 28))
# 第二层
x = keras.layers.Flatten()(input)
x1 = layers.Dense(128, activation='relu')(x)
# 第三层
x2 = layers.Dropout(0.4)(x1)
# 第四层
x3 = layers.Dense(128, activation='relu')(x2)
# 第五层 输出层
output = layers.Dense(10, activation='softmax')(x3)
model = keras.Model(inputs=input, outputs=output)
model.summary()

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['acc']
              )
# 绘图
history = model.fit(x_train, y_train_onehot, epochs=10,
                    validation_data=(x_test, y_test_onehot))
history.history.keys()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.legend()

plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend()
plt.show()
