"""
直接将原图做成模板
"""

import cv2 as cv
import numpy as np
import math
import os

toSize = 128
saveDir = 'Models2/'

def show(name, img) :                        #方便展示图片，方便测试，运行的时候注释掉
    cv.namedWindow(name, cv.WINDOW_NORMAL)    #防止图片太大放不下
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows();

def preHandle(image):  # 去除边框的处理

    gray = cv.GaussianBlur(image, (3, 3), 1)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    ret1, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 使用大基算法，将图像进行二值化
    return thresh



# 将图片保存为txt图像
def save(name, img) :            # 将像素数组放入文件中
    # print("文件保存中...")
    file = open(name, 'w')        # 自动创建文件,如果已经有了则覆盖，区别'x'
    (x, y) = img.shape
    for i in range(x):
        for j in range(y) :
            file.write(str(img[i, j])) #保存为二值图像
        if i != x - 1:      #最后一行不用保存
            file.write('\n')
    file.close()
    # print("保存完成")

# 将TrainImage2下的图片缩小为32 * 32的，然后进行二值化，最后保存到TrainData2中

def changToTxt(saveName, img) :
    img = cv.resize(img, (toSize, toSize), interpolation=cv.INTER_AREA)
    #需要重新进行二值化，因为缩放之后的图像是灰度图，另外转换成0，1更好计算
    # ret1, thresh = cv.threshold(img, 100, 1, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 使用大基算法，将图像进行二值化
    for i in range(len(img)) :
        for j in range(len(img[0])) :
            img[i, j] = 0 if img[i, j] == 255 else 1
    save(saveName, img)


def run() :

    dir = os.listdir('images')

    testTime = len(dir)
    for img_i in range(testTime):
        print("第" + str(img_i) + "张")

        pic = dir[img_i]                                     # 读取图片
        path = 'images/' + pic
        img = cv.imread(path)

        img = preHandle(img)

        saveName = saveDir + pic.split('.')[0] + '.txt'

        changToTxt(saveName, img)       # 原图制作成模板
        print("模板制作完成")


if __name__ == '__main__':
    run()