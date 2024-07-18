import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras

(train_image, train_label), (test_image, test_label) = keras.datasets.fashion_minist.load_data()
train_image = (train_image - 127.5) / 127.5
test_image = (test_image - 127.5) / 127.5
Batch_image = 256
All_size = 60000
datasets = tf.data.Dataset.from_tensor_slices(train_image)
datasets = datasets.shuffle(All_size).batch(Batch_size)


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
    return real_loss, fake_loss


def generator_loss(fake_image):
    return cross_entropy(tf.ones_like(fake_image), fake_image)


generator_opt = tf.keras.optimizers.Adam(learning_rate=0.01)

seed = tf.random.normal([num_exp_to_generate, noise_dim])
generator = generator_model()
discriminator = discriminator_model()


def train_step(images):
    noise = tf.random.normal([Batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_out = discriminator(images, training=True)
        gen_image = generator(noise, training=True)
        fake_out = discriminator(gen_image, training=True)

        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)

    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient()
