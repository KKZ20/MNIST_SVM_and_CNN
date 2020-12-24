import os
import numpy as np
import cv2
from patchify import patchify

'''将图像切割成若干分辨率相同的小图像'''


def image_transfer(image, name, length):
    patches = patchify(image, (length, length, 3), step=length)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            split_image = patches[i, j, 0]
            save_path = os.path.join(
                './myimages/', name + '_{0}_{1}.png'.format(i, j))
            # 将图片转为28*28分辨率并存储
            save_image = cv2.resize(
                split_image, (28, 28), interpolation=cv2.INTER_AREA)
            cv2.imwrite(save_path, save_image)


'''获取自己的手写数字图片'''


def get_mynumbers():
    path = './myimages/whole_image.png'
    name = 'mynumbers'
    image = cv2.imread(path)
    # 截图中的数字以 2 行，6 列的方式排列
    row = 5
    column = 10
    # 首先要保证每一张都是长宽相等的矩形（正方形）
    if (image.shape[0] / row == image.shape[1] / column):
        length = int(image.shape[0] / row)
        # 将自己的图片切割
        image_transfer(image, name, length)


'''将自己的手写数字图片加入到训练/测试集中'''


def load_mynumbers(set_x, set_y):
    # 截图中的数字以 2 行，6 列的方式排列
    row = 5
    column = 10
    label = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                      [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
                      [1, 3, 5, 7, 9, 2, 4, 6, 8, 0],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    for i in range(row):
        for j in range(column):
            path = './myimages/mynumbers' + '_' + \
                str(i) + '_' + str(j) + '.png'
            image = cv2.imread(path)
            # 将图片格式由 RGB 转成灰度图，以符合 MNIST 数据集的格式
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # 将灰度图转换为黑底白字，用 255 - 当前灰度值
            for m in range(28):
                for n in range(28):
                    gray_image[m, n] = 255 - gray_image[m, n]
            # 将灰度图维数改为(1, 28, 28)
            reshape_image = gray_image.reshape(1, 28, 28)
            # 加入训练/测试集
            set_x = np.concatenate((set_x, reshape_image))
            set_y = np.concatenate((set_y, label[i, j].reshape(1)))
    return set_x, set_y
