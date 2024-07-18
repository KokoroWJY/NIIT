# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:17:46 2022

@author: work
"""
import numpy as np
from tensorflow.keras.datasets import imdb  # 数据集
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.preprocessing import sequence  # 序列处理
from tensorflow.keras.optimizers import RMSprop

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000, seed=528)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(x_train[:3])
print(y_train[:3])
length = [len(i) for i in x_train]
np.max(length), np.min(length), np.mean(length)
# 由于每条评论数据长度不一致，需要将每个序列调整为相同的长度
# pad_sequences，将多个序列截断或补齐为相同长度，短于maxlen的默认是前端补齐0， 超过maxlen的，默认是丢弃前面的
x_train_sq = sequence.pad_sequences(x_train, maxlen=500, padding='pre', truncating='pre')
print(x_train_sq.shape)
# print(x_train_sq[5])
# print(x_train[5])
x_test_sq = sequence.pad_sequences(x_test, maxlen=500, padding='pre', truncating='pre')
print(x_test_sq.shape)
model = Sequential()
model.add(Embedding(input_dim=10000,  # 词汇表大小
                    output_dim=32,  # 词向量维度
                    input_length=500  # 输入序列的长度（pad_sequense之后）
                    ))

model.add(SimpleRNN(30, return_sequences=False))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train_sq, y_train, validation_data=(x_test_sq, y_test),
          epochs=5, batch_size=8, verbose=1)
