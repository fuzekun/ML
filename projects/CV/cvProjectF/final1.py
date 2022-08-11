"""
    直接使用分割出来的行进行分割
"""

import cv2 as cv
import numpy as np
import os

toSize = 128
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
    dir = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 左上下，右上下
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
    if flag == 1:                                                                          # 边框操作
        binnay_image = preHandle3(img)
    elif flag == 2:
        ret, binnay_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)       # 第三中


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
    ret, rotatedT = cv.threshold(rotated, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    if flag == 2 :                                                                    # 第二次
        ret, rotatedT = cv.threshold(rotated, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    kernely = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))                         # 腐蚀膨胀
    rotatedT = cv.dilate(rotatedT, kernely)
    rotatedT = cv.morphologyEx(rotatedT, cv.MORPH_OPEN, kernely)
    rotatedT = cv.morphologyEx(rotatedT, cv.MORPH_CLOSE, kernely)

    x = min(min_rect[0])                                                              # 切割
    y = max(min_rect[0])
    height = min(min_rect[1])
    width = max(min_rect[1])
    line = rotatedT[int(x - height / 2):int(x + height / 2),
           max(int(y - width / 2) - 5, 1):min(int(y + width / 2) + 5, int(shape[1]))]

    return line

"""
    1. 读取训练集TainData目录下所有文件和文件夹
    2. 将训练集的数据映射到toSize维的空间中去
    3. 将测试的数字映射到toSize维的空间中去
    4. 选取5个距离最近的排序,选择最近的作为数字
"""
def readTrain(trainingMat, trainingFileList) :                #获取训练集合,返回训练集和标签
    labels = []
    m = len(trainingFileList)

    for i in range(m):
        fileNameStr =  trainingFileList[i]                      # 文件名
        fileStr = fileNameStr.split('.')[0]                     # 前缀名
        labels.append(fileStr)                                  # 将字符存入标签
        returnVect = np.zeros((1, 16384))                        # 创建一个向量
        fr = open('Models/' + fileNameStr, "r")                  # 读取文件
        for j in range(toSize):                                     # 转化为toSize行的向量
            lineStr = fr.readline()
            for k in range(toSize):
                returnVect[0, toSize * j + k] = int(lineStr[k])
        trainingMat[i, :] = returnVect                         # 存入对应的训练集
    return labels

# 将图片缩小为toSize  * toSize 的，然后进行二值化，最后保存到文件中
def changToTxt(img):
    img = cv.resize(img, (toSize, toSize), interpolation=cv.INTER_AREA)
    for i in range(len(img)):
        for j in range(len(img[0])) :
            img[i, j] = 0 if img[i, j] == 255 else 1
    return img


def img2vector(image):
    returnVect = np.zeros((1, toSize * toSize))
    for i in range(toSize):
        line = image[i]
        for j in range(toSize):
            returnVect[0, toSize * i + j] = int(line[j])
    return returnVect

def classify(normData, dataSet, labels, k):
    """

    :param normData: 测试数据
    :param dataSet: 模板数据集合
    :param labels: 标签
    :param k: 取前几名
    :return: 分类结果
    """
    dataSetSize = dataSet.shape[0]              # 模板个数
    diffMat = np.tile(normData, (dataSetSize, 1)) - dataSet # 差分结果
    sqDiffMat = diffMat ** 2                    # 平方
    sqDistances = sqDiffMat.sum(axis=1)         # 求和
    distance = sqDistances ** 0.5               # 开根号
    sortedDistIndicies = distance.argsort()     # 排序提取索引
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]

def IdentifImg(image, name, trainingMat, labels, k):

    image = changToTxt(image)                 # 将图片归一化为toSize  * toSize 的

    vectorUnderTest = img2vector(image)     # 将图片转化为向量
    classifierResult = classify(vectorUnderTest, trainingMat, labels, k) # 分类


    print("识别出来的字符是" + classifierResult)
    print("真实字符是:" + name)
    if name == classifierResult :
        print("正确")
        return True
    else :
        print("错误")
        return False

def run() :

    trainingFileList = os.listdir('Models')
    trainingMat = np.zeros((len(trainingFileList), toSize * toSize))  # m个模板,每个toSize列
    labels = readTrain(trainingMat, trainingFileList)      # 获取训练集，以及对应的标签
    print("训练集读取完成")

    rightT = 0  # 识别正确的字符
    totalNum = 0
    rightS = 0

    dirlist = os.listdir('images')
    testTime = len(dirlist)
    for img_i in range(testTime):
        print("第" + str(img_i) + "张")

        pic = dirlist[img_i]                    # 读取图片
        path = 'images/' + pic
        img = cv.imread(path)
        img = resizeImg(img)                # 图片归一化处理

        line = getLine(img, 0)              # 行分割
        if len(line) == 0:
            line = getLine(img, 1)          # 去除边框在分割
        if len(line) == 0:
            line = getLine(img, 2)          # 反转

        if len(line) == 0:
            print("行分割出错")
            continue

        picName = pic.split('.')[0]  # 前缀
        num = len(picName.split(' ')[1])
        totalNum += num


        right = IdentifImg(line, picName, trainingMat, labels, 1)        # 进行识别

        if right:
            rightT += 1
            rightS += num

    print("总体识别正确次数 %d" % rightT)
    rightRate = rightT / float(testTime)
    print("总体正确率: %f" % rightRate)
    print("单个正确率为: %f" % (rightS / totalNum))

if __name__ == '__main__':
    run()