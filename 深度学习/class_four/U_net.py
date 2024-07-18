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

seed = 7
np.random.seed(seed)

img_w = 256
img_h = 256
train_image_dir = '../img/data_1/imgs(256)'
train_label_dir = '../img/data_1/masks(256)'


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


all_x, all_y = get_train_data(train_image_dir, train_label_dir)
train_x, valid_x, train_y, valid_y = train_test_split(all_x, all_y, test_size=0.25, random_state=seed)


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


img_size_target = 256

# model
input_layer = Input((img_size_target, img_size_target, 1))
output_layer = build_model(input_layer, 16, 0.2)

model = Model(input_layer, output_layer)
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
model.summary()

model_path = r'../img/data_1/result'
if os.path.isdir(model_path):
    pass
else:
    os.makedirs(model_path)
filepath = model_path + '/U.h5'

model_checkpath = ModelCheckpoint(filepath, monitor='accuracy', mode='max', save_best_only=True, verbose=1)

epochs = 5
batch_size = 1
history = model.fit(all_x, all_y, epochs=epochs, batch_size=batch_size, validation_data=(valid_x, valid_y), verbose=2,
                    callbacks=[model_checkpath])

local_path = r'../img/data_1/result/U.h5'
model1 = load_model(local_path)
model1.summary()

test_images_dir = '../img/cut_img'
for parent, dirname, filenames in os.walk(test_images_dir):
    for filename in filenames:
        test_img = cv2.imread(os.path.join(test_images_dir, filename), 0)
        img = np.array(test_img, dtype='float') / 255.0
        img2 = img.reshape(1, 256, 256, 1)
        pred = model.predict(img2)
        a = pred.reshape(65536)
        for i in range(0, 65536):
            if (a[i] < 0.5):
                a[i] = 0
            else:
                a[i] = 1
        pred = a.reshape(256, 256)
        pred = np.array(pred, dtype='float64')
        pred = np.reshape(pred, (img_w, img_h))
        pred1 = pred * 255
        pred1 = np.clip(pred1, 0, 255).astype("float64")

        cv2.imwrite('../img/predict/' + filename + '.png', pred1)
