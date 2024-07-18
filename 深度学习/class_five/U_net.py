#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/27 14:24
# @Author  : Tourior
# @Site    : 
# @File    : U_net.py
# @Software: PyCharm
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

# import cv2
from sklearn.model_selection import train_test_split

# from itertools import chain
from PIL import Image

from keras.models import Model, load_model, save_model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img  # ,save_img

import time

import cv2

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

# img = cv2.imread("29_1.tif",0)/255

# t_start = time.time()

seed = 7
np.random.seed(seed)

img_w = 256
img_h = 256

train_images_dir = 'C:/Users/work/Desktop/1/DRIVE/train/images/'
train_labels_dir = 'C:/Users/work/Desktop/1/DRIVE/train/1st_manual/'


def get_train_data(train_images_dir, train_labels_dir):
    train_images = []
    train_labels = []
    files = os.listdir(train_images_dir)  # get file names
    files2 = os.listdir(train_labels_dir)
    # total_images = np.zeros([len(files), img_h, img_w, N_channels])#for storing training imgs
    for idx in range(len(files)):
        img = cv2.imread(os.path.join(train_images_dir, files[idx]), 1)
        img = img[100:356, 100:356, :]
        # img=np.reshape(img,(img_h, img_w,14))
        img = np.reshape(img, (img_h, img_w, 3))
        #        img = img[:,:,3:12]
        img = np.array(img, dtype="float") / 255.0
        train_images.append(img)

        label = Image.open(os.path.join(train_labels_dir, files2[idx]))
        label = np.array(label, dtype="float") / 255.0
        label = label[100:356, 100:356]
        label = np.reshape(label, (img_h, img_w, 1))
        # label = np.array(label, dtype="float")
        train_labels.append(label)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    return train_images, train_labels


# a = all_y[1]
all_x, all_y = get_train_data(train_images_dir, train_labels_dir)
train_x, valid_x, train_y, valid_y = train_test_split(all_x, all_y, test_size=0.25, random_state=seed)


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.numpy_function(get_iou_vector, [label, pred > 0.5], tf.float64)


def build_model(input_layer, start_neurons, DropoutRatio=0.2):
    # 256 -> 128
    # 第一层卷积，卷积核个数为16
    conv1_1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding="same")(input_layer)
    # conv1_2= Conv2D(start_neurons* 1, (3, 3), activation='relu', padding="same")(conv1_1)
    conv1_2 = residual_block(conv1_1, start_neurons * 1)
    conv1_3 = residual_block(conv1_2, start_neurons * 1, True)
    # pooling层
    pool1 = MaxPooling2D((2, 2))(conv1_3)
    # pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 128 -> 64
    conv2_1 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding="same")(pool1)
    # conv2_2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding="same")(conv2_1)
    conv2_2 = residual_block(conv2_1, start_neurons * 2)
    conv2_3 = residual_block(conv2_2, start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2_3)
    # pool2 = Dropout(DropoutRatio)(pool2)

    # 64 -> 32
    conv3_1 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding="same")(pool2)
    # conv3_2= Conv2D(start_neurons * 4, (3, 3), activation='relu', padding="same")(conv3_1)
    conv3_2 = residual_block(conv3_1, start_neurons * 4)
    conv3_3 = residual_block(conv3_2, start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3_3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # # 12 -> 6
    conv4_1 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4_2 = residual_block(conv4_1, start_neurons * 8)
    conv4_3 = residual_block(conv4_2, start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4_3)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding="same")(pool4)
    # convm = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding="same")(convm)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    # # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4_3])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    # uconv4 = residual_block(uconv4, start_neurons * 8, True)

    # # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3_3])
    # uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    # uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    # # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2_3])

    # uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    # # uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1_3])

    # uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    # uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    # uconv1 = Dropout(DropoutRatio/2)(uconv1)
    # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer


# def get_iou_vector(A, B):
#     batch_size = A.shape[0]
#     metric = []
#     for batch in range(batch_size):
#         t, p = A[batch] > 0, B[batch] > 0
#         intersection = np.logical_and(t, p)
#         union = np.logical_or(t, p)
#         iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
#         thresholds = np.arange(0.5, 1, 0.05)
#         s = []
#         for thresh in thresholds:
#             s.append(iou > thresh)
#         metric.append(np.mean(s))

#     return np.mean(metric)


# def my_iou_metric(label, pred):
#     return tf.py_function(get_iou_vector, [label, pred > 0.5], tf.float64)


img_size_target = 256

# model
input_layer = Input((img_size_target, img_size_target, 3))
output_layer = build_model(input_layer, 16, 0.1)

model1 = Model(input_layer, output_layer)

# c = optimizers.adam(lr = 0.01)
model1.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
               metrics=[my_iou_metric])

model1.summary()

model_path = 'C:/Users/work/Desktop/data/1/'
if os.path.isdir(model_path):
    pass
else:
    os.mkdir(model_path)
filepath = model_path + '/U2-net.h5'
filepath

model_path = "C:/Users/work/Desktop/0515/DRIVE/1"
if os.path.isdir(model_path):
    pass
else:
    os.makedirs(model_path)
filepath = model_path + '/U.h5'

model_checkpath = ModelCheckpoint(filepath, monitor='my_iou_metric', mode='max', save_best_only=True, verbose=1)

# checkpoint = tf.train.Checkpoint(model1)
# checkpoint.save('C:/Users/work/Desktop/data/1/model.h5')

# early_stopping = EarlyStopping(monitor='loss', mode = 'min',patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(filepath, monitor='my_iou_metric',
                                   mode='max', save_best_only=True, verbose=1)

# reduce_lr = ReduceLROnPlateau(monitor='accuracy', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

epochs = 600
batch_size = 8

history = model1.fit(all_x, all_y, validation_data=(valid_x, valid_y),
                     epochs=epochs,
                     batch_size=batch_size, verbose=2, callbacks=[model_checkpoint])
# callbacks=[ model_checkpoint,early_stopping ,reduce_lr],
# verbose=2


model_path = 'C:/Users/work/Desktop/data/1/'

model = load_model(model_path + '/U2-net.h5', custom_objects={'my_iou_metric': my_iou_metric})
#                                                     'lovasz_loss': lovasz_loss}))
model.summary()
# model = load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
#                                                     'lovasz_loss': lovasz_loss})

test_images_dir = 'C:/Users/work/Desktop/1/DRIVE/test/images'
for parent, dirnames, filenames in os.walk(test_images_dir):  # 遍历每一张图片
    for filename in filenames:
        img = cv2.imread(os.path.join(test_images_dir, filename), 1)
        #        img = img[:,:,3:12]
        img = np.array(img, dtype="float") / 255.0
        img = img[100:356, 100:356, :]
        img2 = img.reshape(1, 256, 256, 3)
        pred = model.predict(img2)
        a = pred.reshape(65536)
        for i in range(65536):
            if (a[i] < 0.5):
                a[i] = 0;
            else:
                a[i] = 1
        pred = a.reshape(256, 256)
        pred = np.array(pred, dtype="float64")
        pred = np.reshape(pred, (img_w, img_h))
        pred1 = pred.astype(np.float) * 255.
        pred1 = np.clip(pred1, 0, 255).astype('float64')

        cv2.imwrite(
            "C:/Users/work/Desktop/1/fenge/" + filename + ".png",
            pred1)  # 生成预测结果位置以及文件名

# def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
#     x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
#     preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
#     preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
#     preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
#     return preds_test/2


# preds_valid = predict_result(model,x_valid,img_size_target)
