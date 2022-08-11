'''
    切割图片：
    1. 使用bfs得到每一个图片的recs,如果大于了1.5bottom就直接返回就行了，这样就可以把后面的数字给切割掉
    2. 使用识别进行相应的数字识别。
        2.1 遇见数字就直接比较
        2.2 如果不是数字，直接省略就行。

    3. 可以边切割边识别，从左到右就行。
'''
import math

import cv2 as cv
import numpy as np
from pathlib import Path
import os
def show(name, img) :                        #方便展示图片，方便测试，运行的时候注释掉
    cv.namedWindow(name, cv.WINDOW_NORMAL)    #防止图片太大放不下
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows();

color = 255                            #确定背景图片是0还是1
def transpos(image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 逆时针以图像中心旋转45度
    # - (cX,cY): 旋转的中心点坐标
    # - 45: 旋转的度数，正度数表示逆时针旋转，而负度数表示顺时针旋转。
    # - 1.0：旋转后图像的大小，1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    # OpenCV不会自动为整个旋转图像分配空间，以适应帧。旋转完可能有部分丢失。如果您希望在旋转后使整个图像适合视图，则需要进行优化，使用imutils.rotate_bound.
    M = cv.getRotationMatrix2D((cX, cY), 5, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    # show("Rotated by 5 Degrees", rotated)
    return rotated

def addBounder(image) :                  #给图像增加边框
    (tx, ty) = image.shape
    ret = np.zeros((tx + 10, ty + 10), np.uint8)
    (rx, ry) = (tx + 10, ty + 10)
    for i in range(rx):                                     #补上一圈白色像素
        for j in range(ry):
            if i < 5 or i >= tx + 5 or j < 5 or j >= ty + 5:
                ret[i][j] = 255
            else :
                ret[i][j] = image[i - 5][j - 5]
    return ret
def preHandle(imgPath) :                #图像的预处理
    img = cv.imread(imgPath)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 灰度化
    # 二值化
    ret1, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU ) #使用大基算法，将图像进行二值化
    # save(thresh)

    ret = addBounder(thresh)                   #增加边框
    return ret

# 将存储区域特别小的改成0
def changeTo0(img, tmp) :
    for (x, y) in tmp:
        img[x, y] = 0

# 找联通区域，返回联通区域的最低值
def bfs(grid, x, y, vis, tmpa) :                #tmpa保存内容
    dir = [(-1, 0), (1, 0), (0, -1), (0, 1)    # 上下左右
        , (-1, -1), (1, -1), (-1, 1), (1, 1)]  # 左上下，右上下
    (m, n) = grid.shape
    top = m
    bottom = 0
    left = n
    right = 0
    que = [(x, y)]
    cnt = 1
    while len(que) > 0 :
        (x, y) = que.pop(0)                    #取出队列头
        for (dirx, diry) in dir :
            nx = x + dirx
            ny = y + diry
            if nx >= 0 and nx < m and ny >= 0 and ny < n and vis[nx][ny] == 0 and grid[nx][ny] == 255:
                cnt += 1
                bottom = max(bottom, nx)
                top = min(top, nx)
                left = min(left, ny)
                right = max(right, ny)
                que.append((nx, ny))
                vis[nx][ny] = 1
    tmpa.append(cnt)
    return (top, bottom, left, right)

def findConnection(image, thresh) :                   # 直接分割
    recs = []                                         # 保存所有的字符
    (maxx, maxy) = image.shape
    vis = np.zeros((maxx + 5, maxy + 5))

    cnt = 0
    for i in range(maxx):                               #找到前两个边界值
        flag = 0
        for j in range(maxy):
            if image[i][j] == 255 and vis[i][j] == 0:
                tmpa = []
                rec = bfs(image, i, j, vis, tmpa)           # 连通区域的边界
                # print("lena = " + str(len(tmpa)))
                if tmpa[0] > thresh:                        # 如果不是噪声点
                    if rec[1] > 0.8 * maxy :                # 如果是二维码,底端应该是在最后了
                        flag = 1
                        break
                    # print("top bottom left right:")
                    # for value in rec:
                    #     print(value, end =  " ")
                    # print()
                    # show(str(cnt), image[rec[0]:rec[1], rec[2]:rec[3]])
                    recs.append(rec)
                        # 保存每个字符的边界
            if flag == 1 : break

    recs.sort(key = lambda x: (x[2], x[0]))                 # 按照从左到右,从上到下进行排序
    return recs                                             # 返回每一个联通区域的边界

def readTrain(trainingMat) :                                     #获取训练集合,返回训练集和标签
    labels = []
    trainingFileList = os.listdir('TrainData')
    m = len(trainingFileList)
    # print(m)
    for i in range(m):
        fileNameStr =  trainingFileList[i]                      # 文件名
        fileStr = fileNameStr.split('.')[0]                     # 前缀名
        classNumStr = (fileStr.split('_')[0])                   # 对应的字符
        labels.append(classNumStr)
        returnVect = np.zeros((1, 1024))                        # 创建一个向量
        fr = open('TrainData/' + fileNameStr, "r")              # 读取文件
        for j in range(32):                                     # 转化为1024行的向量
            lineStr = fr.readline()
            for k in range(32):
                returnVect[0, 32 * j + k] = int(lineStr[k])
        trainingMat[i, :] = returnVect                          # 存入对应的训练集
    return labels


def getName(names) :                                         #获取图片数字名称
    namelist = []
    for name in names:
        if name.isdigit():
            namelist.append(name)
    return namelist

if __name__ == '__main__':
    trainingMat = np.zeros((16, 1024))  # 15个模板,每个1024列
    labels = readTrain(trainingMat)  # 获取训练集，以及对应的标签
    print("训练集读取完成")
    pic = "ISBN 978-7-5520-1617-8.JPG"  # 图片名称
    img = preHandle("images/" + pic)
    show('preImage', img)
    prename = pic.split('.')[0]  # 前缀
    names = getName(prename)  # 获取每一个字符实际是多少
    recs = findConnection(img, 50)
    for cnt in range(len(recs)):
        show(str(cnt), img[recs[cnt][0]:recs[cnt][1], recs[cnt][2]: recs[cnt][3]])
