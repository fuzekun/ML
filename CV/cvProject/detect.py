# 设计模板

'''
    1. 读取文件中的模板
    2.

Attendtion：
    1. 每次缩小之后需要重新进行二值化处理
'''

import os
from pathlib import Path
import cv2 as cv
import numpy as np

# 此方法将每个文件中32*32的矩阵数据，转换到1*1024一行中
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

def img2vector(filename):
    # 创建一个1行1024列的矩阵
    returnVect = np.zeros((1, 1024))
    # 打开当前的文件
    fr = open(filename, "rb")
    # 每个3文件中有32行，每行有32列数据，遍历32个行，将2个列数据放入1024的列中
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


"""
    1. 读取训练集TainData目录下所有文件和文件夹
    2. 将训练集的数据映射到1024维的空间中去
    3. 将测试的数字映射到1024维的空间中去
    4. 选取5个距离最近的排序,选择最近的作为数字
"""
def IdentifImg():
    labels = []
    # 读取训练集 TrainData目录下所有的文件和文件夹
    trainingFileList = os.listdir('TrainData')
    m = len(trainingFileList)
    # zeros((m,1024)) 返回一个m行 ，1024列的矩阵，默认是浮点型的
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # 获取文件名称  0_0.txt
        fileNameStr = trainingFileList[i]
        # 获取文件除了后缀的名称
        fileStr = fileNameStr.split('.')[0]
        # 获取文件"字符"的类别
        classNumStr = (fileStr.split('_')[0])
        labels.append(classNumStr)
        # 构建训练集, img2vector  每个文件返回一行数据 1024列
        trainingMat[i, :] = img2vector('TrainData/%s' % fileNameStr)
    # 读取测试集数据
    testFileList = os.listdir('TestData')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i] #0_0.txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = (fileStr.split('_')[0])
        vectorUnderTest = img2vector('TestData/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, labels, 5)
        print("识别出的字符是: %c, 真实字符是: %c" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\n识别错误次数 %d" % errorCount)
    errorRate = errorCount / float(mTest)
    print("\n正确率: %f" % (1 - errorRate))


# 将图片保存为txt图像
def save(name, img) :            # 将像素数组放入文件中
    print("文件保存中...")
    file = open(name, 'w')        # 自动创建文件,如果已经有了则覆盖，区别'x'
    (x, y) = img.shape
    for i in range(x):
        for j in range(y) :
            file.write(str(img[i, j])) #保存为二值图像
        if i != x - 1:      #最后一行不用保存
            file.write('\n')
    print("保存完成")

# 将图片缩小为32 * 32的，然后进行二值化，最后保存到文件中
def changToTxt() :
    list = ['X']
    for i in range(len(list)):
        name = 'TrainImages/' + list[i] + '.jpg'
        myfile = Path(name)
        if myfile.is_file():                                        #如果是文件的话
            img = cv.imread(name, 0)
            img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
            #需要重新进行二值化，因为缩放之后的图像是灰度图，另外转换成0，1更好计算
            ret1, thresh = cv.threshold(img, 100, 1, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 使用大基算法，将图像进行二值化
            save('TrainData/' + list[i] + '_1.txt', thresh)
            cv.imshow('img' + list[i], thresh)
    cv.waitKey()
    cv.destroyAllWindows()

if __name__ == '__main__':
    changToTxt()
    # IdentifImg()