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
            if (area / areaT > 0.03 and area / areaT < 0.07) and (min__rect[1][0] + min__rect[1][1] > 900):
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
    # kernely = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))       # 获取核
    # binnary = cv.dilate(binnary, kernely)                           # 腐蚀
    # binnary = cv.morphologyEx(binnary, cv.MORPH_OPEN, kernely)      # 膨胀
    # binnary = cv.morphologyEx(binnary, cv.MORPH_CLOSE, kernely)
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
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
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

def getColumn2(image, thresh) :                  # 使用bfs进行列切割(上到下，左到右)
    """

    :param image: 行
    :param thresh: 分割的阈值
    :return: 每一个分割后的数字，字符等
    """
    (maxx, maxy) = image.shape
    vis = np.zeros((maxx + 5, maxy + 5))
    recs = []                                   # 保存每一个图像的边界
    for i in range(maxx):                                  # 找到前两个边界值
        for j in range(maxy):
            if image[i][j] == 255 and vis[i][j] == 0:
                tmpa = []
                rec = bfs2(image, i, j, vis, tmpa)          # 连通区域的边界
                # print("lena = " + str(len(tmpa)))
                if tmpa[0] > thresh:                        # 如果不是噪声点
                    recs.append(rec)                        # 保存每个字符的边界
    recs.sort(key = lambda x:x[2])                          # 根据列进行排序                                #
    imgs = []
    cnt = 0                                               # 展示计数
    for rec in recs:                                        # 得到每一列
        img = image[rec[0]:rec[1], rec[2]:rec[3]]
        imgs.append(img)
        # show(str(cnt), img)
        cnt += 1
    return imgs


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
"""
    1. 读取训练集TainData目录下所有文件和文件夹
    2. 将训练集的数据映射到1024维的空间中去
    3. 将测试的数字映射到1024维的空间中去
    4. 选取5个距离最近的排序,选择最近的作为数字
"""
def readTrain(trainingMat, path) :                             #获取训练集合,返回训练集和标签
    labels = []
    trainingFileList = os.listdir(path)
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


# 将图片缩小为32 * 32的，然后进行二值化，最后保存到文件中
def changToTxt(image):  # 返回一个32 * 32的二值图
    img = cv.resize(image, (32, 32), interpolation=cv.INTER_AREA)
    ret1, thresh = cv.threshold(img, 100, 1, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return thresh



def img2vector(image):
    # 创建一个1行1024列的矩阵
    returnVect = np.zeros((1, 1024))
    # 每个图像中有32行，每行有32列数据，遍历32个行，将2个列数据放入1024的列中
    for i in range(32):
        line = image[i]
        for j in range(32):
            returnVect[0, 32 * i + j] = int(line[j])
    return returnVect

def classify(normData, dataSet, labels, k):
    # 计算行数
    dataSetSize = dataSet.shape[0]
    #     print ('dataSetSize 长度 =%d'%dataSetSi  ；                  vzvz ze)
    # 当前点到所有点的坐标差值  ,np.tile(x,(y,1)) 复制x 共y行 1列
    diffMat = np.tile(normData, (dataSetSize, 1)) - dataSet
    # 对每个坐标差值平方
    sqDiffMat = diffMat ** 2
    # 对于二维数组 sqDiffMat.sum(axis=0)指 对向量每列求和，sqDiffMat.sum(axis=1)是对向量每行求和,返回一个长度为行数的数组
    # 例如：narr = array([[ 1.,  4.,  6.],
    #                   [ 2.,  5.,  3.]])
    #    narr.sum(axis=1) = array([ 11.,  10.])
    #    narr.sum(axis=0) = array([ 3.,  9.,  9.])
    sqDistances = sqDiffMat.sum(axis=1)
    # 欧式距离 最后开方
    distance = sqDistances ** 0.5
    # x.argsort() 将x中的元素从小到大排序，提取其对应的index 索引，返回数组
    # 例：   tsum = array([ 11.,  10.])    ----  tsum.argsort() = array([1, 0])
    sortedDistIndicies = distance.argsort()
    #     classCount保存的K是魅力类型   V:在K个近邻中某一个类型的次数
    classCount = {}
    for i in range(k):
        # 获取对应的下标的类别
        voteLabel = labels[sortedDistIndicies[i]]
        # 给相同的类别次数计数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # sorted 排序 返回新的list
    #     sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]

def IdentifImg(images, names, trainingMat, labels, totalNum):

    right = 0

    mTest = len(images)                     # 有几张图片
    expectList = ['I', 'S', 'B', 'N', '-']

    print('识别出字符是:')

    cnt = 0                                   #当前是第几章图片
    for i in range(mTest):
        img = images[i]

        img = changToTxt(img)                 # 将图片归一化为32 * 32的

        vectorUnderTest = img2vector(img)     # 将图片转化为向量
        classifierResult = classify(vectorUnderTest, trainingMat, labels, 5)  # 分类
        if classifierResult in expectList:    # 如果是ISBN这种字符,不进行识别
            continue

        print("%c" % classifierResult, end="")
        if (classifierResult == names[cnt]):  #如果按照顺序可以识别出来
            right += 1.0
        if cnt < totalNum - 1: cnt += 1       # 防止识别过多的数字出错

    print("\n真实字符是:")
    for name in names:
        if name not in expectList:
            print(name, end="")
    print()
    return right                                             # 第一个是顺序识别出来的个数,

def getName(names) :                                         # 获取图片数字名称
    namelist = []
    for name in names:
        if name.isdigit() or name == 'X':
            namelist.append(name)
    return namelist

def run() :

    trainingMat = np.zeros((17, 1024))                      # 15个模板,每个1024列
    labels = readTrain(trainingMat, "TrainData")           # 获取训练集，以及对应的标签
    print("训练集读取完成")

    errorT = 0              # 总部错误
    rightT = 0              # 识别正确的字符
    total = 0               # 总字符的个数
    error_cnt = 0           # 分割出错的

    dir = os.listdir('images')
    file = open('wronglne.txt', 'w')
    fw = open('wrongName.txt', 'w')

    testTime = len(dir)
    for img_i in range(testTime):
        print("第" + str(img_i) + "张")

        pic = dir[img_i]                    # 读取图片
        path = 'images/' + pic
        img = cv.imread(path)

        line = getLine(img)                 # 行分割
        if len(line) == 0:
            print("行分割出错")
            file.write(dir[img_i])
            file.write("\n")
            error_cnt += 1
            errorT += 1
            total += totalNum
            continue

        imgs = getColumn(line)              # 列分割

        prename = pic.split('.')[0]             # 前缀

        names = getName(prename)                # 获取正确的列表

        totalNum = len(names)

        right = IdentifImg(imgs, names, trainingMat, labels, totalNum)          # 进行识别

        if right != totalNum:
            fw.write(pic)
            fw.write('\n')
            errorT += 1
        print("正确个数为:%d" % right)
        rightT += right             # 识别正确的个数
        total += totalNum           # 总的识别次数


    fw.close()
    file.close()

    print("总体识别错误次数 %d" % errorT)
    errorRate = errorT / float(testTime)
    print("总体正确率: %f" % (1 - errorRate))
    rightRate = rightT / float(total)
    print("单个正确率: %f" % rightRate)



if __name__ == '__main__':
    run()







