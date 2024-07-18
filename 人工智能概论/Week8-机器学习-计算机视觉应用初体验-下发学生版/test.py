import tensorflow_hub as hub

hub_handle = 'https://hub.tensorflow.google.cn/google/magenta/arbitrary-image-stylization-v1-256/2'
# # 加载模型
hub_model = hub.load(hub_handle)
print(hub_model)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



def mix_iamge(face_image):
    content_image = plt.imread(r'D:\Python\python code\NIIT_class\人工智能概论\Week8-机器学习-计算机视觉应用初体验-下发学生版\img\ori.jpg')
    style_image = plt.imread(
        r'D:\Python\python code\NIIT_class\人工智能概论\Week8-机器学习-计算机视觉应用初体验-下发学生版\img\{}.jpg'.format(face_image))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(content_image)

    plt.subplot(1, 2, 2)
    plt.imshow(style_image)
    plt.show()
    # 图像标准化，数字转化为0-1之间,增加一个维度，形成4维张量（样本数，长，宽，通道数）
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    # 风格图最好转换为（256，256）大小，与训练模型的风格图大小是（256，256）
    # 原图可随意
    style_image = tf.image.resize(style_image, (256, 256))
    # 展示正方形处理后的风格图
    plt.imshow(style_image[0])
    plt.show()

    # 风格迁移
    # 之前加载好的模型：hub_model. tf.constant: 生成一个常量张量
    outputs = hub_model(tf.constant(content_image), tf.constant(style_image))

    # 绘图展现
    plt.figure(figsize=(16, 6))

    # 原图像
    plt.subplot(1, 2, 1)
    plt.imshow(content_image[0])
    plt.title('original image')

    # 风格图像
    plt.subplot(1, 2, 2)
    plt.imshow(style_image[0])
    plt.title('style image')
    plt.show()
    # 迁移后图像
    plt.imshow(outputs[0][0])
    plt.title('stylized image')
    plt.show()
