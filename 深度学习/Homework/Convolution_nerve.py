""" 卷积神经网络 作业 """
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

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

x_train_4d = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 整形后的4维数据 x_train_4d
x_test_4d = x_test.reshape(x_test.shape[0], 28, 28, 1)  # 整形后的4维数据  x_test_4d


def get_model_API(input_layer):
    # conv
    conv1 = Conv2D(32, (3, 3), activation='relu', padding="same")(input_layer)
    # conv
    conv2 = Conv2D(64, (3, 3), activation='relu', padding="same")(conv1)
    # pooling
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv
    conv3 = Conv2D(64, (3, 3), activation='relu', padding="same")(pool)
    # conv
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    # dropout
    dropout = Dropout(0.2)(conv4)
    # flatten
    flatten = Flatten()(dropout)
    # dense
    conv5 = Dense(512, activation='relu')(flatten)
    # dense
    conv6 = Dense(128, activation='relu')(conv5)
    # out
    output_layer = Dense(10, activation='softmax')(conv6)
    return output_layer


img_size_target = 28
input_layer = Input((img_size_target, img_size_target, 1))
output_layer = get_model_API(input_layer)
model = Model(input_layer, output_layer)
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


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


hs_cnn = model.fit(x_train_4d, y_train,
                   batch_size=20, epochs=20,
                   validation_data=(x_test_4d, y_test)
                   )
plot_loss(hs_cnn)
