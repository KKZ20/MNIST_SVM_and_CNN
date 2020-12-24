import numpy as np
import os
import pickle
import tensorflow as tf
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm

import ImageDivide

'''步骤一：下载手写数字数据集，进行初步的数据可视化和统计'''

# 载入MNIST官方数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # 下载数据集

# 载入自己的数据集
# imageDivide.get_mynumbers()
# 载入进训练集
# x_train, y_train = image.load_mynumbers(x_train, y_train)
# 载入进测试集
x_test, y_test = ImageDivide.load_mynumbers(x_test, y_test)


# 可视化MNIST训练集的前9张图片
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(331 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.savefig('./information_images/sample_MNIST.png',
            bbox_inches='tight', dpi=300)

# 可视化自己制作数字训练集的前9张图片
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(331 + i)
    plt.imshow(x_test[i + 10000], cmap=plt.get_cmap('gray'))
plt.savefig('./information_images/sample_MINE.png',
            bbox_inches='tight', dpi=300)

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
plt.savefig('./information_images/label_distribution.png',
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


# ------------------SVM Start---------------------
'''步骤三：模型搭建'''
print(time.strftime('%Y-%m-%d %H:%M:%S'))
clf = svm.SVC(C=100.0, kernel='rbf', gamma=0.03, verbose=True)

# 输出模型信息
print('模型信息')
print(clf.get_params())

'''步骤四：训练模型'''

t1 = time.time()
if os.path.exists('./model/svm.pkl'):
    with open('./model/svm.pkl', 'rb') as f1:
        clf = pickle.load(f1)

else:
    clf.fit(x_train, y_train)
    with open('./model/svm.pkl', 'wb') as f2:
        pickle.dump(clf, f2)

t2 = time.time()
SVMfit = float(t2-t1)
print("训练时间: {} seconds".format(SVMfit))

# ------------------SVM end-----------------------

'''步骤五：评估模型'''
print('模型测试')

predictions = [int(a) for a in clf.predict(x_test)]
# 混淆矩阵
print(confusion_matrix(y_test, predictions))
# f1-score,precision,recall
print(classification_report(y_test, np.array(predictions)))

# 各个参数变化
print('accuracy: ', accuracy_score(y_test, predictions))
