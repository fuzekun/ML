import cv2
import cv2 as cv
import numpy as np
import os

def show(name, img) :                        #方便展示图片，方便测试，运行的时候注释掉
    cv.namedWindow(name, cv.WINDOW_NORMAL)    #防止图片太大放不下
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows();

#得到字符区域
def get_letter_area(contours, copyimg):
    for i in range(0, len(contours)):
        min__rect = cv.minAreaRect(contours[i])
        if min(min__rect[1]) > 1:
            if float(max(min__rect[1]) / min(min__rect[1])) <= 23 and float(max(min__rect[1]) / min(min__rect[1])) >= 12 and \
                    min__rect[1][0] + min__rect[1][1] > 500:
                res = cv.drawContours(copyimg, contours[i], -1, (0, 255, 0), 4)
                return i

# 图像预处理

def preHandle(img) :
    gray = cv.GaussianBlur(img, (5, 5), 1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, binnay_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (120, 1))
    kernely = cv.getStructuringElement(cv.MORPH_RECT, (2, 15))
    binnay = cv.morphologyEx(binnay_image, cv2.MORPH_CLOSE, kernelX)  # 形态学操作
    binnay = cv.morphologyEx(binnay, cv2.MORPH_OPEN, kernely)
    # show('binnary', binnay)
    return (binnay, binnay_image)

def get_revolve_angle():              #求旋转角度
    if min_rect[2]<-45:
        return min_rect[2]+90
    else :
        return min_rect[2]



def cutLine(min_rect, rotated) :                    # 进行图片的行分割
    x = min(min_rect[0])
    y = max(min_rect[0])
    height = min(min_rect[1])
    width = max(min_rect[1])
    cutimg = rotated[int(x - height / 2) - 2:int(x + height / 2) + 2,
             int(y - width / 2) - 5:int(y + width / 2) + 5]
    cutimg_height, cutimg_width = cutimg.shape
    cnt = [0] * cutimg_width                        # 统计当这一行有多少个字符
    for i in range(0, cutimg_width):
        for j in range(0, cutimg_height):
            if cutimg[j][i] == 255:
                cnt[i] = cnt[i] + 1
    mean = []
    a = [0, cutimg_width - 1]
    for i in range(0, cutimg_width - 1):
        if (cnt[i] == 0 and cnt[i + 1] > 0) or (i == 0 and cnt[0] > 0):
            a[0] = i
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

    for k in range(0, len_):
        recs = []
        recutimg = cutimg[0:cutimg_height, mean[k][0]:mean[k][1]]
        recs.append(recutimg)
        # show(str(k), recutimg)
    return recs

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

def img2vector(image):
    # 创建一个1行1024列的矩阵
    returnVect = np.zeros((1, 1024))
    # 每个图像中有32行，每行有32列数据，遍历32个行，将2个列数据放入1024的列中
    for i in range(32):
        line = image[i]
        for j in range(32):
            returnVect[0, 32 * i + j] = int(line[j])
    return returnVect

"""
    1. 读取训练集TainData目录下所有文件和文件夹
    2. 将训练集的数据映射到1024维的空间中去
    3. 将测试的数字映射到1024维的空间中去
    4. 选取5个距离最近的排序,选择最近的作为数字
"""
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


# 将图片缩小为32 * 32的，然后进行二值化，最后保存到文件中
def changToTxt(image) :              # 返回一个32 * 32的二值图
    img = cv.resize(image, (32, 32), interpolation = cv.INTER_AREA)
    ret1, thresh = cv.threshold(img, 100, 1, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return thresh

def IdentifImg(images, names, trainingMat, labels, totalNum):     # 测试数据的图像对应数字以及训练集
    right = 0
    rightCount = 0
    mTest = len(images)
    expectList = ['I', 'S', 'B', 'N', '-']
    print('识别出字符是:')
    cnt = 0                                         # 当前的数字
    for i in range(mTest):
        img = images[i]
        # show(str(i), img)
        img = changToTxt(img)                           # 将图片归一化为32 * 32的
        # print('TestData/' + names[i] + '_1.txt')
        # save('TestData/' + names[i] + '_1.txt', img)
        vectorUnderTest = img2vector(img)               # 将图片转化为向量
        classifierResult = classify(vectorUnderTest, trainingMat, labels, 5)    # 分类
        if classifierResult in expectList :
            continue
        print("%c" % classifierResult, end = "")
        if classifierResult in names:
            rightCount += 1
        if (classifierResult == names[cnt]):
            right += 1.0
        if cnt < totalNum - 1: cnt += 1                #防止识别过多的数字出错
    print("\n真实字符是:")
    for name in names:
        if name not in expectList:
            print(name, end="")
    rightRate = rightCount / float(totalNum)
    print()
    # print("\n正确率: %f" % rightRate)
    return (right, rightCount)                          # 第一个是顺序识别出来的个数,
                                                        # 第二个是无需识别出来的个数


# 将图片保存为txt图像
def save(name, img) :            # 将像素数组放入文件中
    print("文件保存中...")
    file = open(name, 'w')        # 自动创建文件,如果已经有了则覆盖，区别'x'
    (x, y) = img.shape
    for i in range(x):
        for j in range(y) :
            file.write(str(1 if img[i][j] > 0 else 0)) #保存为二值图像
        if i != x - 1:      #最后一行不用保存
            file.write('\n')
    print("保存完成")



def getName(names) :                                         #获取图片数字名称
    namelist = []
    for name in names:
        if name.isdigit() or name == 'X':
            namelist.append(name)
    return namelist



def printTrain(trainingMat) :                               #输出训练集
    for line in trainingMat:
        for v in line:
            print(v, end = " ")
        print()



if __name__ == '__main__':
    trainingMat = np.zeros((16, 1024))  # 15个模板,每个1024列
    labels = readTrain(trainingMat)  # 获取训练集，以及对应的标签
    print("训练集读取完成")

    errorT = 0  # 全部错误
    rightT = 0  # 识别正确的字符
    total = 0  # 总字符的个数
    trainingFileList = os.listdir('images')  # 读取images下的文件
    m = len(trainingFileList)
    fw = open('wrongName.txt', 'w')  # 打开文件，写入错误的名称和错误率
    # fw2 = open('wrongln', 'w')                          #写入错误的分割数量的图片
    for i in range(m):
        if i == 1 or i == 2 or i == 4: continue
        print('第%d章图片' % i)
        pic = trainingFileList[i]
        img = cv.imread('images/' + pic)
        (binary, binay_img) = preHandle(img)  # 得到轮廓和二值化之后的图像
        copyimg = img  # 由于会改变原来的图像，所以暂时保存现在的图像
        wer, contours, abc = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        id = get_letter_area(contours, copyimg)
        min_rect = cv2.minAreaRect(contours[id])  # 返回最小外接矩形
        box = cv.boxPoints(min_rect)
        box = np.int0(box)
        cv.drawContours(copyimg, [box], 0, (0, 0, 255), 4)
        shape = binary.shape
        angle = get_revolve_angle()  # 获取旋转角度
        M = cv.getRotationMatrix2D(min_rect[0], angle, 1.0)  # 得到图像旋转角度
        rotated = cv.warpAffine(binay_img, M, (shape[1], shape[0]))
        imgs = cutLine(min_rect, rotated)
        prename = pic.split('.')[0]  # 前缀
        names = getName(prename)  # 获取每一个字符实际是多少
        # print('imgs.len = %d, names.len = %d' % (len(imgs), len(names)))
        totalNum = len(names)
        if len(imgs) != len(names):
            print("分割数量出错")
            # fw2.write(pic)
            # fw2.write('\n')

        (right, rightCount) = IdentifImg(imgs, names, trainingMat, labels, totalNum)
        if right != totalNum:
            fw.write(pic)
            fw.write('\n')
            errorT += 1
        print("正确个数为:%d" % rightCount)
        rightT += rightCount  # 识别正确的个数
        total += totalNum  # 总的识别次数


    fw.close()
    # fw2.close()
    print("总体识别错误次数 %d" % errorT)
    errorRate = errorT / float(m)
    print("总体正确率: %f" % (1 - errorRate))
    rightRate = rightT / float(total)
    print("单个正确率: %f" % rightRate)



