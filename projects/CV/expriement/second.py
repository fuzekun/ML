# 使用灰度阈值分割，静态阈值分割105，区分背景和前景。
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gray_cvt(inputimagepath):  # 使用cv2默认的方法灰度化
    img = cv2.imread(inputimagepath)
    gray_cvt_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 灰度化
    # cv2.imwrite(outimagepath, gray_cvt_image)  # 保存当前灰度值处理过后的文件
    return gray_cvt_image

def changeToArray() :               #转化为数组
    img = gray_cvt('images/ISBN 978-7-122-29990-1.png')
    image = np.array(img)
    value = []
    for line in image:
        value.append(line)
    return value



def draw() : #绘制灰度直方图
    value = changeToArray()
    # 1.统计每一个灰度值的
    val = []
    num = np.zeros(256)
    for line in value:
        for v in line:
            val.append(v)
            num[v] += 1
    # 自动统计value中的每一个值有多少
    plt.plot(range(256), num)
    plt.hist(val, bins = 256, rwidth=2, color='yellow')
    plt.savefig('pixel.png')  #保存图像
    plt.show()



def rgb2gray(rgb):
    # 将rgb转灰度
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def otsu(gray) : #使用大基算法得到最佳的阈值分割点

    #    return 0.2989 * image[:,:,0] + 0.5870*image[:,:,1] + 0.1140*image[:,:,2]
    h, x = np.histogram(gray.ravel(), bins=256)

    plt.imshow(gray, cmap=plt.get_cmap('gray'))  # 显示灰度图
    plt.show()

    # 整体均值
    meanall = np.sum(np.dot(h, np.array([n for n in range(256)])))
    meanall = meanall / np.sum(np.array([n for n in range(256)]))
    maxscore = 0;
    gi = []
    for i in range(1, 255):
        # 分割图像的均值只需要将直方图的高度乘以x坐标求和，再除以x坐标之和
        mean1 = np.sum(np.dot(h[0:i], np.array([n for n in range(i)])))
        mean1 = mean1 / np.sum(h[0:i])
        mean2 = np.sum(np.dot(h[i:256], np.array([n for n in range(i, 256)])))
        mean2 = mean2 / np.sum(h[i:256])
        # 公式计算
        score = sum(h[0:i]) * ((meanall - mean1) ** 2) + sum(h[i:256]) * ((meanall - mean2) ** 2)

        #    用于绘图
        gi.append(score)
        if maxscore < score:  # 记录最大值
            maxscore = score
            threshold = i

    # print("max value = %d, th = %d" % (maxscore, threshold))

    # 用于绘图
    gi.append(min(gi))
    gi.insert(0, min(gi))
    plot1 = plt.figure()
    # 绘制直方图
    plt.bar(np.array([n for n in range(256)]), h)
    # 绘制分割点
    plt.axvline(threshold, color='r')
    # 绘制类间方差遍历过程示意图
    plt.scatter([n for n in range(256)], (gi - min(gi)) / (max(gi) - min(gi)) * max(h))
    plt.show()
    return threshold


'''
1. 先将图片灰度化处理
2. 将处理后的图片进行选择阈值之后进行固定阈值全局分割，前景为0，背景为255
3. 分割之后
'''

def doThreshold(img, thresh) : #实际实现分割的函数
    ret = np.zeros((len(img), len(img[0])), np.uint8)
    for i in range(len(img)) :
        for j in range(len(img[0])) :
            if img[i][j] > thresh :
                ret[i][j] = 255
            else :
                ret[i][j] = 0
    return ret
def myThreshold() :
    # 1.灰度化处理
    image = gray_cvt('images/ISBN 978-7-122-29990-1.png')
    # 2. 使用大基算法得到阈值
    thresh = otsu('images/ISBN 978-7-122-29990-1.png')

    # 3. 使用阈值进行固定阈值分割图像
    img = doThreshold(image, thresh)
    cv2.imshow("besthresh", img)

    # 4. 使用肉眼观察的进行对比
    img = doThreshold(image, 100)
    cv2.imshow("thresh", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    draw()              #绘制二分图像
    myThreshold()       #进行灰度化的处理