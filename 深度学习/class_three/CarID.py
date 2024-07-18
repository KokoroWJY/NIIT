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
import matplotlib.pyplot as plt
from keras.layers.merge import concatenate
import tensorflow as tf
import numpy as np

cifar10 = tf.keras.datasets.cifar10
(train_image, train_label), (test_image, test_label) = cifar10.load_data()
plt.imshow(train_image[1])
plt.show()

np.max(train_image[0])
np.min(train_image[0])
train_image = train_image / 255.0
test_image = test_image / 255.0


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


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


def get_model_cnn_API(input_layer):
    conv1_1 = Conv2D(8, (3, 3), activation='relu', padding="same")(input_layer)
    # conv1_2  = BatchActivate(conv1_1)
    # inception结构块
    conv2_1 = Conv2D(8, (1, 1), activation='relu', padding="same")(conv1_1)

    conv2_2 = Conv2D(8, (1, 1), activation='relu', padding="same")(conv1_1)
    conv2_2 = Conv2D(8, (3, 3), activation='relu', padding="same")(conv2_2)

    conv2_3 = Conv2D(8, (1, 1), activation='relu', padding="same")(conv1_1)
    conv2_3 = Conv2D(8, (5, 5), activation='relu', padding="same")(conv2_3)

    conv2_4 = concatenate([conv2_3, conv2_2, conv2_1])

    conv1_2 = Conv2D(24, (3, 3), activation=None, padding="same")(conv2_4)
    conv1_2 = BatchActivate(conv1_2)
    conv1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    # conv11 = Conv2D(64,(3,3),strides=(2,2),activation=None,padding="same")(conv10)
    conv1_3 = Conv2D(64, (3, 3), activation='relu', padding="same")(conv1_2)
    conv1_4 = Conv2D(64, (3, 3), activation=None, padding="same")(conv1_3)
    conv1_4 = BatchActivate(conv1_4)
    conv1_5 = MaxPooling2D(pool_size=(2, 2))(conv1_4)
    # conv1_5  = Add()([conv1_2, conv1_4])

    conv1_5 = Flatten()(conv1_5)
    # conv1_5  = Dropout(0.2)(conv1_5)
    conv1_6 = Dense(512, activation='relu')(conv1_5)
    conv1_7 = Dense(128, activation='relu')(conv1_6)
    output_layer = Dense(10, activation='softmax')(conv1_7)
    return output_layer
