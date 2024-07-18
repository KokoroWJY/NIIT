def inception_(input_layer):
    conv1 = Conv2D(16, (1, 1), activation='relu', padding='same')(input_layer)
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

    conv2 = Conv2D(16, (1, 1), activation='relu', padding='same')(input_layer)
    conv2_1 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv2)

    max_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv_max_pool = Conv2D(32, (1, 1), activation='relu', padding='same')(max_pool)

    conv = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    concatenation = concatenate([conv1_1, conv2_1, conv_max_pool, conv])

    return concatenation


def residual_block(Input):
    conv1 = convolution_block(Input, 64, (1, 1))
    conv2 = convolution_block(conv1, 64, (3, 3))
    conv3 = convolution_block(conv2, 256, (1, 1), activation=False)
    conv4 = BatchNormalization()(conv3)
    conv5 = Add()([conv4, Input])
    conv6 = Activation('relu')(conv5)
    return conv6
