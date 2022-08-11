import os

import cv2 as cv
import numpy as np

def show(name, img) :                        #方便展示图片，方便测试，运行的时候注释掉
    cv.namedWindow(name, cv.WINDOW_NORMAL)    #防止图片太大放不下
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows();


def get_chars_area(areaT, contours):
    """

    :param areaT: 二值化图片
    :param contours: 联通区域
    :return: 得到的联通区域的id
    """
    for i in range(0, len(contours)) :
        min__rect = cv.minAreaRect(contours[i])
        if min(min__rect[1])>5:
            area = min__rect[1][0] * min__rect[1][1]
            if (area / areaT > 0.03 and area / areaT < 0.07) :
                # print('area = ' + str(area))
                # print('比例:%f' % (area / areaT))
                return i

def resize_img(img) :
    # 1. 等比例缩放图片大小
    height, width = img.shape[0:2]
    img = cv.resize(img, (int(width * 1400 / height), int(1400)))
    return img

def mphlog(binnay_image):
    """

    :param binnay_image: 二值图
    :return:             膨胀之后
    """
    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (100, 1))
    kernely = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    binnay = cv.morphologyEx(binnay_image, cv.MORPH_CLOSE, kernelX)  # 形态学操作
    binnay = cv.morphologyEx(binnay, cv.MORPH_OPEN, kernely)
    return binnay

def getChars(areaT, contImg) :
    """

    :param areaT:   缩放后图像米面积
    :param contImg: 膨胀后的图片
    :return:        字符的联通区域
    """
    wer, contours, t = cv.findContours(contImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    idx = get_chars_area(areaT, contours)

    if idx == None:
        return []
    return contours[idx]

def get_degree(min_rect):              #求旋转角度
    if min_rect[2]<-45:
        return min_rect[2]+90
    else :
        return min_rect[2]

def transpos(min_rect, gray) :
    """

    :param contour:  字符联通区域
    :param gray: 灰度图
    :return :    旋转之后的图片
    """
    shape = gray.shape
    degree = get_degree(min_rect)
    M = cv.getRotationMatrix2D(min_rect[0], degree, 1.0)
    rotated = cv.warpAffine(gray, M, (shape[1], shape[0]))
    return rotated

def cutLine(img, min_rect) :
    """

    :param img: 旋转之后的灰度图
    :param min_rect: 最小外接矩形
    :return: 切出来的一行
    """
    ret, binnary = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernely = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))       # 获取核
    binnary = cv.dilate(binnary, kernely)                           # 腐蚀
    binnary = cv.morphologyEx(binnary, cv.MORPH_OPEN, kernely)      # 膨胀
    binnary = cv.morphologyEx(binnary, cv.MORPH_CLOSE, kernely)
    x = min(min_rect[0])  # 中心
    y = max(min_rect[0])
    height = min(min_rect[1])                                         # 外界矩形的宽高
    width = max(min_rect[1])
    shape = binnary.shape
    line = binnary[int(x - height / 2) - 2:int(x + height / 2) + 2,
             max(int(y - width / 2) - 5, 1):min(int(y + width / 2) + 5, int(shape[1]))]
    return line

def getLine(img) :
    """

    :param img: 图片
    :return: 行
    """
    img = resize_img(img)  # 图片缩放之后
    gray=cv.GaussianBlur(img,(3,3),1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binnay_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 图像灰度二值化

    contImg = mphlog(binnay_image)  # 膨胀后

    contour = getChars(len(img) * len(img[0]), contImg)

    if (len(contour) == 0): return []

    min_rect = cv.minAreaRect(contour)  # 图片的最小外界矩形

    rotated = transpos(min_rect, gray)  # 旋转

    line = cutLine(rotated, min_rect)  # 行分割

    return line


def bfs2(grid, x, y, vis, cnts) :                       #cnts保存数量
    dir = [(-1, 0), (1, 0), (0, -1), (0, 1)  # 上下左右
        , (-1, -1), (1, -1), (-1, 1), (1, 1)]  # 左上下，右上下
    (m, n) = grid.shape
    top = m
    bottom = 0
    left = n
    right = 0
    que = [(x, y)]
    cnt = 0
    while len(que) > 0 :
        (x, y) = que.pop(0)                    #取出队列头
        for (dirx, diry) in dir :
            nx = x + dirx
            ny = y + diry
            if nx >= 0 and nx < m and ny >= 0 and ny < n and vis[nx][ny] == 0 and grid[nx][ny] == 255:
                bottom = max(bottom, nx)
                top = min(top, nx)
                left = min(left, ny)
                right = max(right, ny)
                que.append((nx, ny))
                cnt += 1
                vis[nx][ny] = 1
    cnts.append(cnt)
    return (top, bottom, left, right)



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
                mean.append(a)

    len_ = len(mean)

    imgs = []
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
        imgs.append(recutimg1)
    return imgs

def getColumn2(image, thresh) :                  # 使用bfs进行列切割(上到下，左到右)
    """

    :param image: 行
    :param thresh: 分割的阈值
    :return: 每一个分割后的数字，字符等
    """
    (height, width) = image.shape
    cnt = [0] * width
    for i in range(width) :
        for j in range (height):
            if image[j][i] == 255:
                cnt[i] += 1
    borders = []
    a = [0, width - 1]
    for i in range(width - 1) :
        if (cnt[i] == 0 and cnt[i + 1] > 0) or (i == 0 and cnt[0] > 0):
           a[0] = i + 1
        if (cnt[i] > 0 and cnt[i + 1] > 0) :
            a[1] = i
            if a[1] - a[0] > 7:
                borders.append(a)
                a = [0, width - 1]
        if i == width - 2 > 0 and a[0] != 0:
            a[1] = i
            if (a[1] - a[0] > 7) :
                borders.append(a)
                a = [0, width - 1]

    imgs = []
    for i in range(len(borders)):
        column = image[0:height, borders[i][0]:borders[i][1]]
        top = 0
        bottom = height
        (h, w) = column.shape
        flag = 0
        for i in range(h) :
            for j in range(w) :
                if column[i][j] == 255 :
                    top = i
                    flag = 1
                    break
            if flag == 1:
                break
        flag = 0
        for i in range(1, h) :
            for j in range(w):
                if (column[height - i][j] == 255) :
                    bottom = height - i
                    flag = 1
                    break
            if flag == 1:
                break
        img = column[top:bottom]
        imgs.append(img)
    return imgs



def getName(names) :                                         # 获取图片数字名称
    namelist = []
    for name in names:
        if name.isdigit() or name == 'X':
            namelist.append(name)
    return namelist

def run() :

    pic = 'ISBN 978-0-12-811953-2.jpg'                    # 读取图片
    path = 'images/' + pic
    img = cv.imread(path)

    line = getLine(img)                 # 行分割
    show('line', line)

    if len(line) == 0:
        print("行分割出错")

    imgs = getColumn(line)              # 列分割

    for i in range(len(imgs)):
        show(str(i), imgs[i])


if __name__ == '__main__':
    run()







