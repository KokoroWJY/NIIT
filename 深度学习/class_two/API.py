# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:10:49 2022

@author: Lucifer
"""

from tensorflow.keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    Dropout, Conv2DTranspose, Add
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, load_model, save_model
from tensorflow.keras import Sequential  # 顺序模型
from tensorflow.keras.layers import Flatten, Dense, Conv2D  # 全连接层，2D卷积层
from tensorflow.keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 数据加载
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train_4d = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 整形后的4维数据 x_train_4d
x_test_4d = x_test.reshape(x_test.shape[0], 28, 28, 1)  # 整形后的4维数据  x_test_4d


# 定义BN层
def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


# 定义cnn_API
def get_model_cnn_API(input_layer):
    conv10 = Conv2D(32, (3, 3), activation=None, padding="same")(input_layer)
    conv10 = BatchActivate(conv10)
    # conv11 = Conv2D(64,(3,3),strides=(2,2),activation=None,padding="same")(conv10)
    conv11 = Conv2D(64, (3, 3), activation=None, padding="same")(conv10)
    conv12 = Flatten()(conv11)
    conv13 = Dense(128, activation='relu')(conv12)
    output_layer = Dense(10, activation='softmax')(conv13)
    return output_layer


img_size_target = 28
input_layer = Input((img_size_target, img_size_target, 1))
output_layer = get_model_cnn_API(input_layer)

model = Model(input_layer, output_layer)
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
hs_cnn = model.fit(x_train_4d, y_train,
                   batch_size=20, epochs=10,
                   validation_data=(x_test_4d, y_test)
                   )
