def get_model_cnn_API(input_layer):
    conv1_1 = Conv2D(8, (3, 3), activation='relu', padding="same")(input_layer)

    # inception结构块
    conv2_1 = Conv2D(8, (1, 1), activation='relu', padding="same")(conv1_1)

    conv2_2 = Conv2D(8, (1, 1), activation='relu', padding="same")(conv1_1)
    conv2_2 = Conv2D(8, (3, 3), activation='relu', padding="same")(conv2_2)

    conv2_3 = Conv2D(8, (1, 1), activation='relu', padding="same")(conv1_1)
    conv2_3 = Conv2D(8, (5, 5), activation='relu', padding="same")(conv2_3)

    conv2_4 = concatenate([conv2_3, conv2_2, conv2_1])

    # 卷积
    conv1_2 = Conv2D(24, (3, 3), activation=None, padding="same")(conv2_4)
    conv1_2 = BatchActivate(conv1_2)
    conv1_3 = Conv2D(64, (3, 3), activation='relu', padding="same")(conv1_2)

    # 全连接
    conv1_4 = Flatten()((conv1_3))
    conv1_5 = Dense(512, activation='relu')(conv1_4)
    conv1_6 = Dense(128, activation='relu')(conv1_5)
    output_layer = Dense(10, activation='softmax')(conv1_6)
    return output_layer