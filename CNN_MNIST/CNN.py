# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import image

'''步骤一：下载手写数字数据集，进行初步的数据可视化和统计'''

# 载入MNIST官方数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # 下载数据集

# 载入自己的数据集
image.get_mynumbers()
# 载入进训练集
# x_train, y_train = image.load_mynumbers(x_train, y_train)
# 载入进测试集
# x_test, y_test = image.load_mynumbers(x_test, y_test)


# 可视化MNIST训练集的前9张图片
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(331 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.savefig('./infomation_images/sample.png', bbox_inches='tight', dpi=300)

# 训练集和测试集的部分信息
print(f'训练集的样本数：{x_train.shape[0]}，测试集的样本数：{x_test.shape[0]}')
print(f'输入图像的大小：{x_train.shape[1]}*{x_train.shape[2]}')

# 训练集中各类标签的扇形分布图
label_cnt = Counter(y_train)  # 统计
print('训练集的图像类别分布：', label_cnt)
plt.figure(figsize=(5, 5))
plt.pie(x=label_cnt.values(),
        labels=label_cnt.keys(),
        autopct='%.2f%%')
plt.savefig('./infomation_images/label_distribution.png',
            bbox_inches='tight', dpi=300)

'''步骤二：数据预处理'''
pixel_size = x_train.shape[1] * x_test.shape[2]  # 图像尺寸
num_class = len(label_cnt)  # 标签数量

# 将 28x28 的图像展平成一个 784 维的向量
x_train = x_train.reshape(x_train.shape[0], pixel_size)
x_test = x_test.reshape(x_test.shape[0], pixel_size)

x_train = x_train.astype('float32')  # 数据类型转化
x_test = x_test.astype('float32')
x_train /= 255  # 规范化，将像素值缩至0-1之间
x_test /= 255

# CNN训练，转换数据格式(samples, rows, cols, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 标签转换为独热码（one-hot）
y_train = tf.keras.utils.to_categorical(y_train, num_class)
y_test = tf.keras.utils.to_categorical(y_test, num_class)


'''步骤三：模型搭建'''
model = tf.keras.models.Sequential()  # Sequential顺序模型

# 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
model.add(tf.keras.layers.Conv2D(
    32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# 第2层卷积，卷积核大小为3*3，64个
model.add(tf.keras.layers.Conv2D(
    64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# 第3层卷积，卷积核大小为3*3，64个
model.add(tf.keras.layers.Conv2D(
    64, (3, 3), activation='relu', padding='same'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.summary()  # 查看模型架构信息
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])  # 定义模型训练细节，包括交叉熵损失函数，Adam优化器和准确率评价指标

'''步骤四：训练模型'''
# 60000张训练图片， batch_size为128，训练5个epoch
h = model.fit(x_train, y_train, batch_size=128,
              epochs=5, validation_data=(x_test, y_test))

'''步骤五：评估模型'''
print('模型测试')
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f'测试集损失值: {test_loss}, 测试集准确率: {test_acc}')

'''步骤六：模型训练可视化'''
# 混淆矩阵显示
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)

# 各个参数变化
print(h.history.keys())
accuracy = h.history['accuracy']
val_accuracy = h.history['val_accuracy']
loss = h.history['loss']
val_loss = h.history['val_loss']
epochs = range(len(accuracy))

# 准确率变化图
plt.figure()
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'bo', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('./infomation_images/accuracy.png', bbox_inches='tight', dpi=300)

# 损失值变化图
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./infomation_images/loss.png', bbox_inches='tight', dpi=300)
