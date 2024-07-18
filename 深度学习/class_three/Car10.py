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

    # inception结构块
    conv2_1 = Conv2D(8, (1, 1), activation='relu', padding="same")(conv1_1)

    conv2_2 = Conv2D(8, (1, 1), activation='relu', padding="same")(conv1_1)
    conv2_2 = Conv2D(8, (3, 3), activation='relu', padding="same")(conv2_2)

    conv2_4 = concatenate([conv2_2, conv2_1])

    # 卷积
    conv1_2 = Conv2D(24, (3, 3), activation=None, padding="same")(conv2_4)
    conv1_2 = BatchActivate(conv1_2)
    conv1_3 = Conv2D(64, (3, 3), activation='relu', padding="same")(conv1_2)

    # 全连接
    conv1_4 = Flatten()((conv1_3))
    conv1_6 = Dense(128, activation='relu')(conv1_4)
    output_layer = Dense(10, activation='softmax')(conv1_6)
    return output_layer


img_size_target = 32
input_layer = Input((img_size_target, img_size_target, 3))
output_layer = get_model_cnn_API(input_layer)
model = Model(input_layer, output_layer)
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
hs_cnn = model.fit(train_image, train_label,
                   batch_size=20, epochs=5,
                   validation_data=(test_image, test_label)
                   )
plot_loss(hs_cnn)
