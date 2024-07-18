from tensorflow.keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    Dropout, Conv2DTranspose, Add

from tensorflow.keras import Sequential  # 顺序模型
from tensorflow.keras.layers import Flatten, Dense, Conv2D  # 全连接层，2D卷积层
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

model = Sequential()
model.add(Conv2D(filters=999,
                 kernel_size=(3, 3),  # 卷积核大小
                 input_shape=(6, 6, 1),  # 图片大小 默认valid表示不填充。same表示填充
                 strides=1,  # 步长
                 padding='same'  # 填充
                 ))
model.summary()
# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train的shape
x_train.shape
# y_train的shape
y_train.shape
# 数据归一化
x_train = x_train / 255
x_test = x_test / 255
# 输出维度
x_train[0].shape
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.figure(figsize=(10, 2))


def plot_loss(hs):
    plt.figure(figsize=(8, 4))
    plt.plot(hs.history['loss'], label='Train')
    plt.plot(hs.history['val_loss'], label='test')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(loc=0)
    plt.show()
    return


x_train_4d = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 整形后的4维数据 x_train_4d
x_test_4d = x_test.reshape(x_test.shape[0], 28, 28, 1)  # 整形后的4维数据  x_test_4d
print(x_train_4d.shape, x_test_4d.shape)

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten


# 构建一个CNN结构，并编译。

def get_model_cnn():
    model = Sequential()

    # 增加卷积层, 卷积核20个，大小3*3， 输入图片是28*28
    model.add(Conv2D(filters=20, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu', padding='same'))

    # 池化层：最大池化：2*2
    model.add(MaxPool2D(pool_size=(2, 2)))

    # 展开到一维
    model.add(Flatten())

    # 全连接层，128节点
    model.add(Dense(128, activation='relu'))

    # 输出层，10节点
    model.add(Dense(10, activation='softmax'))

    # 编译：随机梯度下降学习率0.1，损失函数分类交叉熵
    model.compile(optimizer=SGD(learning_rate=0.1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 生成模型
cnn_model = get_model_cnn()
cnn_model.summary()
hs_cnn = cnn_model.fit(x_train_4d, y_train,
                       batch_size=20, epochs=10,
                       validation_data=(x_test_4d, y_test)
                       )
plot_loss(hs_cnn)
