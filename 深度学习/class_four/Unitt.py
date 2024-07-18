from sklearn.model_selection import train_test_split

from keras.models import Model, load_model, save_model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import array_to_img, img_to_array, load_img  # ,save_img

import tensorflow as tf

import cv2
import os
import numpy as np

# get_need_data() 所需要的
train_images_dir = r'../img/data_1/imgs/'
train_labels_dir = r'../img/data_1/masks'
train_images1_dir = r'../img/data_1/imgs(256)/'
train_labels1_dir = r'../img/data_1/masks(256)/'

train_images_dir = r'../img/need_cut_img'
train_images1_dir = r'../img/cut_img/'


def get_need_data(train_images_dir, train_labels_dir):
    ''' 照片获取边角 '''
    train_images = []
    train_label = []
    files = os.listdir(train_images_dir)
    files2 = os.listdir(train_labels_dir)
    for idx in range(len(files)):
        img = cv2.imread(os.path.join(train_images_dir, files[idx]), 0)
        img_1 = img[0:256, 0:256]
        cv2.imwrite(os.path.join(train_images1_dir, '1_' + files[idx]), img_1)
        img_2 = img[0:256, 256:512]
        cv2.imwrite(os.path.join(train_images1_dir, '2_' + files[idx]), img_2)
        img_3 = img[256:512, 0:256]
        cv2.imwrite(os.path.join(train_images1_dir, '3_' + files[idx]), img_3)
        img_4 = img[256:512, 256:512]
        cv2.imwrite(os.path.join(train_images1_dir, '4_' + files[idx]), img_4)

    for idx2 in range(len(files2)):
        lab = cv2.imread(os.path.join(train_labels_dir, files2[idx2]), 0)
        lab_1 = lab[0:256, 0:256]
        cv2.imwrite(os.path.join(train_labels1_dir, '1_' + files2[idx2]), lab_1)
        lab_2 = lab[0:256, 256:512]
        cv2.imwrite(os.path.join(train_labels1_dir, '2_' + files2[idx2]), lab_2)
        lab_3 = lab[256:512, 0:256]
        cv2.imwrite(os.path.join(train_labels1_dir, '3_' + files2[idx2]), lab_3)
        lab_4 = lab[256:512, 256:512]
        cv2.imwrite(os.path.join(train_labels1_dir, '4_' + files2[idx2]), lab_4)


# get_train_data所需要的
seed = 7
np.random.seed(seed)

img_w = 256
img_h = 256
# train_image_dir = '../img/data_1/imgs(256)'
# train_label_dir = '../img/data_1/masks(256)'


def get_train_data(train_images_dir, train_labels_dir):
    train_images = []
    train_labels = []
    files = os.listdir(train_images_dir)
    files2 = os.listdir(train_labels_dir)
    for idx in range(len(files)):
        img = cv2.imread(os.path.join(train_images_dir, files[idx]), 0)
        img = np.reshape(img, (img_h, img_w, 1))
        img = np.array(img, dtype='float') / 255.0
        train_images.append(img)

        label = cv2.imread(os.path.join(train_labels_dir, files2[idx]), 0)
        label = np.reshape(label, (img_h, img_w, 1))
        label = np.array(label, dtype='float') / 255.0
        train_labels.append(label)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    return train_images, train_labels


# all_x, all_y = get_train_data(train_image_dir, train_label_dir)
# train_x, valid_x, train_y, valid_y = train_test_split(all_x, all_y, test_size=0.25, random_state=seed)


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = BatchActivate(x)
    return x


def build_model(input_layer, start_neurons, DropoutRatio=0.5):
    # 第一层
    conv1_1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D(2, 2)(conv1_2)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 第二层
    conv2_1 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D(2, 2)(conv2_2)
    pool2 = Dropout(DropoutRatio / 2)(pool2)

    # 第三层
    conv3_1 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(conv3_1)
    pool3 = MaxPooling2D(2, 2)(conv3_2)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 补
    conv4_1 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4_2 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(conv4_1)
    pool4 = MaxPooling2D(2, 2)(conv4_2)
    pool4 = Dropout(DropoutRatio)(pool4)

    # 中间层
    convm_1 = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same')(pool4)
    convm_2 = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same')(convm_1)

    # 补
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding='same')(convm_2)
    uconv4 = concatenate([deconv4, conv4_2])
    uconv4_1 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv4)
    uconv4_2 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv4_1)

    # 反卷积, 第一次
    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding='same')(uconv4_2)
    uconv3 = concatenate([deconv3, conv3_2])
    uconv3_1 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv3)
    uconv3_2 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv3_1)

    # 第二次
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding='same')(uconv3_2)
    uconv2 = concatenate([deconv2, conv2_2])
    uconv2_1 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(uconv2)
    uconv2_2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(uconv2_1)

    # 第三次
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding='same')(uconv2_2)
    uconv1 = concatenate([deconv1, conv1_1])
    uconv1_1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(uconv1)
    uconv1_2 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(uconv1_1)

    output_layer_noActi = Conv2D(1, (1, 1), padding='same', activation=None)(uconv1_2)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    return output_layer


# img_size_target = 256
#
# # model
# input_layer = Input((img_size_target, img_size_target, 1))
# output_layer = build_model(input_layer, 16, 0.2)
#
# model = Model(input_layer, output_layer)
# model.compile(loss='binary_crassentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
# model.summary()


# 作业小测试
def test_U_plus(input_layer, start_neurons):
    # 卷积
    x_0_0 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(input_layer)

    x_1_0 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(x_0_0)
    x_1_0 = MaxPooling2D((2, 2))(x_1_0)

    x_2_0 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(x_1_0)
    x_2_0 = MaxPooling2D((2, 2))(x_2_0)

    # 反卷积
    dx_1_0 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding='same')(x_1_0)
    x_0_1 = concatenate([x_0_0, dx_1_0])

    dx_2_0 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding='same')(x_2_0)
    x_1_1 = concatenate([dx_2_0, x_1_0])

    dx_1_1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding='same')(x_1_1)
    x_0_2 = concatenate([dx_1_1, x_0_1])

    out_layer = Activation('sigmoid')(x_0_2)
    return out_layer


# img_size_target = 256
#
# # model
# input_layer = Input((img_size_target, img_size_target, 1))
# output_layer = test_U_plus(input_layer, 16, 0.2)
#
# model = Model(input_layer, output_layer)
# model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
# model.summary()

if __name__ == '__main__':
    get_need_data(train_images_dir, train_labels_dir)
