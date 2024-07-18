# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:23:11 2022

@author: work
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:37:45 2022

@author: work
"""
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras

print(tf.__version__)
# from tensorflow import keras
# 加载数据集
(train_image, train_label), (test_image, test_label) = keras.datasets.mnist.load_data()
# train_image的shape
train_image.shape
# train_label的shape
train_label.shape
# 数据归一化
train_image = (train_image - 127.5) / 127.5
test_image = (test_image - 127.5) / 127.5
Batch_size = 256
train_image = train_image.reshape(train_image.shape[0], 28, 28, 1)
All_size = 60000
datasets = tf.data.Dataset.from_tensor_slices(train_image)
datasets
datasets = datasets.shuffle(All_size).batch(Batch_size)
datasets


def generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(100,), use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(512, input_shape=(100,), use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(28 * 28 * 1, use_bias=False, activation='tanh'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Reshape((28, 28, 1)))

    return model


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(512, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(256, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_image, fake_image):
    real_loss = cross_entropy(tf.ones_like(real_image), real_image)
    fake_loss = cross_entropy(tf.zeros_like(fake_image), fake_image)
    return real_loss + fake_loss


def generator_loss(fake_image):
    return cross_entropy(tf.ones_like(fake_image), fake_image)


generator_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

Epochs = 100
noise_dim = 100
# 生成一些可观察的数据集
num_exp_to_generate = 16
# 给16个长度为100的随机数
seed = tf.random.normal([num_exp_to_generate, noise_dim])

generator = generator_model()
discriminator = discriminator_model()


# 定义批次训练函数

def train_step(images):
    noise = tf.random.normal([Batch_size, noise_dim])
    # 自定义训练，使用GradientTape()来记录梯度
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_out = discriminator(images, training=True)
        gen_image = generator(noise, training=True)
        fake_out = discriminator(gen_image, training=True)

        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)
    # 记录变量与损失之间的梯度
    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # 使用梯度优化变量
    generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))


# 定义可视化
def generate_plot_image(gen_model, test_noise):
    pre_images = gen_model(test_noise, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(pre_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((pre_images[i, :, :, 0] + 1) / 2, cmap='gray')
        # 灰度图不显示坐标
        plt.axis('off')
    plt.show()


def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
            print('.', end='')
        generate_plot_image(generator, seed)


train(datasets, Epochs)
