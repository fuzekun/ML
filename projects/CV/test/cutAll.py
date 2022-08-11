"""
    1. 将图片进行行列分割
        1.1 膨胀腐蚀的试一试
        1.2 不做的试一次

"""
import os

import cv2 as cv
import numpy as np
import math

def show(name, img) :                        #方便展示图片，方便测试，运行的时候注释掉
    cv.namedWindow(name, cv.WINDOW_NORMAL)    #防止图片太大放不下
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows();


def addBounder(image):
    """
    给图像加上边框
    :param image: 原图
    :return: None
    """
    (tx, ty) = image.shape
    ret = np.zeros((tx + 10, ty + 10), np.uint8)
    (rx, ry) = (tx + 10, ty + 10)
    for i in range(rx):  # 补上一圈白色像素
        for j in range(ry):
            if i < 5 or i >= tx + 5 or j < 5 or j >= ty + 5:
                ret[i][j] = 255
            else:
                ret[i][j] = image[i - 5][j - 5]
    return ret


def bfs(grid, x, y):
    """
    bfs去除边框
    :param grid:  图像
    :param x: 起使坐标
    :param y:
    :return: None
    """
    dir = [(-1, 0), (1, 0), (0, -1), (0, 1)  # 上下左右
        , (-1, -1), (1, -1), (-1, 1), (1, 1)]  # 左上下，右上下
    (m, n) = grid.shape
    que = [(x, y)]
    while len(que) > 0:
        (x, y) = que.pop(0)  # 取出队列头
        for (dirx, diry) in dir:
            nx = x + dirx
            ny = y + diry
            if nx >= 0 and nx < m and ny >= 0 and ny < n and grid[nx][ny] == 255:
                grid[nx][ny] = 0
                que.append((nx, ny))


def preHandle3(image):  # 去除边框的处理

    gray = cv.GaussianBlur(image, (3, 3), 1)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    ret1, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 使用大基算法，将图像进行二值化
    ret = addBounder(thresh)  # 增加边框
    bfs(ret, 0, 0)  # 去除边框
    return ret


def getLetterArea(contours):
    """
    获取字符的联通区域
    :param contours: 图片所有的联通区域
    :return:         祖父联通区域
    """
    for i in range(0, len(contours)):
        min__rect = cv.minAreaRect(contours[i])
        if min(min__rect[1]) > 5:
            if float(max(min__rect[1]) / min(min__rect[1])) <= 25 and float(
                    max(min__rect[1]) / min(min__rect[1])) >= 7 and \
                    min__rect[1][0] + min__rect[1][1] > 900:
                return i


def resizeImg(img):
    """
    等比例缩放图片大下
    :param img: 原图
    :return: 缩放后的图片
    """
    height, width = img.shape[0:2]
    img = cv.resize(img, (int(width * 1400 / height), int(1400)))
    return img


def getAngel(min_rect):
    """
    求旋转角度
    :param min_rect:    最小外界矩形
    :return:            旋转角度
    """
    if min_rect[2] < -45:
        return min_rect[2] + 90
    else:
        return min_rect[2]


def getLine(img, flag):
    """
    图片进行行分割
    :param img: 归一化后的图片
    :return: 分割的行
    """

    gray = cv.GaussianBlur(img, (3, 3), 1)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    ret, binnay_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值化
    if flag:                                                                          # 边框操作
        binnay_image = preHandle3(img)

    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (100, 1))                       #腐蚀膨胀
    kernely = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    binnay = cv.morphologyEx(binnay_image, cv.MORPH_CLOSE, kernelX)
    binnay = cv.morphologyEx(binnay, cv.MORPH_OPEN, kernely)

    wer, contours, abc = cv.findContours(binnay, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # 找连通区域

    idx = getLetterArea(contours)

    if idx == None: return []

    min_rect = cv.minAreaRect(contours[idx])                                          # 区域最小外界矩形

    shape = gray.shape                                                                # 旋转
    angle = getAngel(min_rect)
    M = cv.getRotationMatrix2D(min_rect[0], angle, 1.0)
    rotated = cv.warpAffine(gray, M, (shape[1], shape[0]))
    ret, rotated = cv.threshold(rotated, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # kernely = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))                         # 腐蚀膨胀
    # rotated = cv.dilate(rotated, kernely)
    # rotated = cv.morphologyEx(rotated, cv.MORPH_OPEN, kernely)
    # rotated = cv.morphologyEx(rotated, cv.MORPH_CLOSE, kernely)

    x = min(min_rect[0])                                                              # 切割
    y = max(min_rect[0])
    height = min(min_rect[1])
    width = max(min_rect[1])
    line = rotated[int(x - height / 2):int(x + height / 2),
           max(int(y - width / 2) - 5, 1):min(int(y + width / 2) + 5, int(shape[1]))]

    return line


def getColumn(cutimg) :
    cutimg_height, cutimg_width = cutimg.shape
    cnt = [0] * cutimg_width

    for i in range(0, cutimg_width):
        for j in range(0, cutimg_height):
            if cutimg[j][i] == 255:
                cnt[i] = cnt[i] + 1

    mean = []
    a = [0, cutimg_width - 1]
    for i in range(0, cutimg_width - 1):
        if (cnt[i] == 0 and cnt[i + 1] > 0) or (i == 0 and cnt[0] > 0):
            a[0] = i + 1
        if cnt[i] > 0 and cnt[i + 1] == 0:
            a[1] = i
            if a[1] - a[0] > 7:
                mean.append(a)
                a = [0, cutimg_width - 1]
        if i == cutimg_width - 2 > 0 and a[0] != 0:
            a[1] = i
            if a[1] - a[0] > 7:
                mean.append(a)

                a = [0, cutimg_width - 1]

    len_ = len(mean)
    ret = []
    for k in range(0, len_):
        recutimg = cutimg[0:cutimg_height, mean[k][0]:mean[k][1]]
        m = 0
        p = cutimg_height
        height, width = recutimg.shape
        flag = 0

        for i in range(0, height):
            for j in range(0, width):
                if recutimg[i][j] == 255:
                    m = i
                    flag = 1
                    break
            if flag == 1:
                break
        flag = 0
        for i in range(1, height):
            for j in range(0, width):
                if recutimg[cutimg_height - i][int(j)] == 255:
                    p = cutimg_height - i
                    flag = 1
                    break
            if flag == 1:
                break

        recutimg1 = cutimg[m:p, mean[k][0]:mean[k][1]]
        ret.append(recutimg1)
    return ret


def getName(names) :                                         # 获取图片数字名称
    namelist = []
    for name in names:
        if name != ' ':
            namelist.append(name)
    return namelist

def run() :

    error_line = 0           # 分割出错的
    errorC = 0

    dir = os.listdir('images')
    file = open('wronglne.txt', 'w')
    file2 = open('wrongCne.txt', 'w')

    testTime = len(dir)
    for img_i in range(testTime):
        print("第" + str(img_i) + "张")

        pic = dir[img_i]                    # 读取图片
        path = 'images/' + pic
        img = cv.imread(path)
        img = resizeImg(img)                   # 图片归一化处理

        line = getLine(img, 0)                 # 行分割
        if len(line) == 0:
            line = getLine(img, 1)             # 去除边框在分割
        if len(line) == 0:
            print("行分割出错")
            file.write(dir[img_i])
            file.write("\n")
            error_line += 1
            continue

        imgs = getColumn(line)  # 列分割
        prename = pic.split('.')[0]  # 前缀
        names = getName(prename)  # 获取正确的列表
        totalNum = len(names)

        print(str(len(imgs)) + " " + str(totalNum))

        # 判断是否裂分割出错
        if totalNum - 4 == len(imgs):  # 去掉空之后的全部
            totalNum -= 4
            names = names[4:]

        if (len(imgs) != totalNum):
            print("列分割出错")
            errorC += 1
            file2.write(pic)
            file2.write('\n')
    file.close()
    file2.close()

    print("列分割错误:" + str(errorC))
    print("行分割错误" + str(error_line))




if __name__ == '__main__':
    run()







