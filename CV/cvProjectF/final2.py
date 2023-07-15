"""
    直接使用分割出来的行进行分割
"""

import cv2 as cv
import numpy as np
import os

toSize = 128
TrainList = 'Models2/'


def preHandle(image):  # 去除边框的处理

    gray = cv.GaussianBlur(image, (3, 3), 1)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    ret1, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 使用大基算法，将图像进行二值化
    return thresh

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
        returnVect = np.zeros((1, toSize * toSize))                        # 创建一个向量
        fr = open(TrainList + fileNameStr, "r")                  # 读取文件
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
    trainingFileList = os.listdir(TrainList)
    trainingMat = np.zeros((len(trainingFileList), toSize * toSize))  # m个模板,每个toSize列
    labels = readTrain(trainingMat, trainingFileList)      # 获取训练集，以及对应的标签
    print("训练集读取完成")

    rightT = 0              # 识别正确的字符
    totalNum = 0
    rightS = 0

    dirlist = os.listdir('images')
    testTime = len(dirlist)
    for img_i in range(testTime):
        print("第" + str(img_i) + "张")

        pic = dirlist[img_i]                    # 读取图片
        path = 'images/' + pic
        img = cv.imread(path)

        img = preHandle(img)

        picName = pic.split('.')[0]             # 前缀
        num = len(picName.split(' ')[1])
        totalNum += num

        right = IdentifImg(img, picName, trainingMat, labels, 1)        # 进行识别

        if right:
            rightT += 1
            rightS += num


    print("总体识别正确次数 %d" % rightT)
    rightRate = rightT / float(testTime)
    print("总体正确率: %f" % rightRate)
    print("单个正确率为: %f" % (rightS / totalNum))

if __name__ == '__main__':
    run()