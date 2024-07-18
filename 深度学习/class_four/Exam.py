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

from PIL import Image
from PIL import ImageSequence

train_image_dir = r'../img/test/DRIVE/train/images/'
train_label_dir = r'../img/test/DRIVE/train/1st_manual/'
test_image_dir = r'../img/test/DRIVE/test/images/'
test_label_dir = r'../img/test/DRIVE/test/1st_manual/'

img_w = 64
img_h = 64
seed = 7


def get_train_data(train_images_dir, train_labels_dir):
    train_images = []
    train_labels = []
    files = os.listdir(train_images_dir)
    files2 = os.listdir(train_labels_dir)
    for idx in range(len(files)):
        img = cv2.imread(os.path.join(train_images_dir, files[idx]), 1)
        img = img[196:260, 196:260, :]
        img = np.reshape(img, (img_h, img_w, 3))
        img = np.array(img, dtype='float') / 255.0
        train_images.append(img)

        label = Image.open(os.path.join(train_labels_dir, files2[idx]))
        label = np.array(label, dtype='float') / 255.0
        label = label[196:260, 196:260]
        label = np.reshape(label, (img_h, img_w, 1))
        train_labels.append(label)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    return train_images, train_labels


def build_model(input_layer, start_neurons, DropoutRatio=0.5):
    # 第一层
    conv1_1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D(2, 2)(conv1_2)
    # pool1 = Dropout(DropoutRatio / 2)(pool1)

    # 第二层
    conv2_1 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D(2, 2)(conv2_2)
    # pool2 = Dropout(DropoutRatio / 2)(pool2)

    # 第三层
    conv3_1 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(conv3_1)
    pool3 = MaxPooling2D(2, 2)(conv3_2)
    # pool3 = Dropout(DropoutRatio)(pool3)

    # 补
    conv4_1 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4_2 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(conv4_1)
    pool4 = MaxPooling2D(2, 2)(conv4_2)
    # pool4 = Dropout(DropoutRatio)(pool4)

    conv5_1 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(pool4)
    conv5_2 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(conv5_1)
    pool5 = MaxPooling2D(2, 2)(conv5_2)

    # 中间层
    convm_1 = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same')(pool5)
    convm_2 = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same')(convm_1)

    deconv5 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding='same')(convm_2)
    uconv5 = concatenate([deconv5, conv5_2])
    uconv5_1 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv5)
    uconv5_2 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv5_1)

    # 补
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding='same')(uconv5_2)
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


all_x, all_y = get_train_data(train_image_dir, train_label_dir)
train_x, valid_x, train_y, valid_y = train_test_split(all_x, all_y, test_size=0.25, random_state=seed)


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


if __name__ == '__main__':
    img_size_target = 64
    input_layer = Input((img_size_target, img_size_target, 3))
    output_layer = build_model(input_layer, 16, 0.2)

    model = Model(input_layer, output_layer)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=[my_iou_metric])
    model.summary()

    model_path = r'../img/data_1/result'
    if os.path.isdir(model_path):
        pass
    else:
        os.makedirs(model_path)
    filepath = model_path + '/U_Exam.h5'

    model_checkpath = ModelCheckpoint(filepath, monitor=my_iou_metric, mode='max', save_best_only=True, verbose=1)

    epochs = 1000
    batch_size = 8
    history = model.fit(all_x, all_y, epochs=epochs, batch_size=batch_size, validation_data=(valid_x, valid_y),
                        verbose=2,
                        callbacks=[model_checkpath])

    # local_path = r'../img/data_1/result/U_Exam.h5'
    # model = load_model(local_path)
    # model.summary()

    for parent, dirname, filenames in os.walk(test_image_dir):
        for filename in filenames:
            test_img = cv2.imread(os.path.join(test_image_dir, filename), 1)
            img = np.array(test_img, dtype='float') / 255.0
            img = img[196:260, 196:260]
            img2 = img.reshape(1, 64, 64, 3)
            pred = model.predict(img2)
            a = pred.reshape(4096)
            for i in range(0, 4096):
                if (a[i] < 0.5):
                    a[i] = 0
                else:
                    a[i] = 1
            pred = a.reshape(64, 64)
            pred = np.array(pred, dtype='float64')
            pred = np.reshape(pred, (img_w, img_h, 1))
            pred1 = pred * 255
            pred1 = np.clip(pred1, 0, 255).astype("float64")

            cv2.imwrite('../img/predict/' + filename + '.png', pred1)