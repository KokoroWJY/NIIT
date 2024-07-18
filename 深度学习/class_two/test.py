from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Conv2D  # 全连接层，2D卷积层
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(rotation_range=20,  # 随机翻转角度 0-100
                             width_shift_range=0.05,  # 随机水平偏移比例
                             height_shift_range=0.05,  # 随机水平垂直比例
                             shear_range=0.1,  # 随机错切换角度
                             zoom_range=0.1,  # 随机缩放范围
                             horiznotal_flip=False,  # 随机水平翻转
                             fill_mode='nearest'  # 填充新像素方法
                             )
# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 数据归一化
x_train = x_train / 255
x_test = x_test / 255
x_train_4d = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 整形后的4维数据 x_train_4d
x_test_4d = x_test.reshape(x_test.shape[0], 28, 28, 1)  # 整形后的4维数据  x_test_4d
datagen.fit(x_train_4d)

gen_data = datagen.flow(x_train_4d, y_train, batch_size=20)

fig, ax = plt.subplot(nrows=1, ncols=10, figsize=(4, 8))
