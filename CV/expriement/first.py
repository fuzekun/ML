import cv2
import numpy as np


def gray_cvt(inputimagepath, windowname): #使用cv2默认的方法灰度化
    img = cv2.imread(inputimagepath)
    gray_cvt_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 灰度化
    # cv2.imwrite(outimagepath, gray_cvt_image)  # 保存当前灰度值处理过后的文件
    return gray_cvt_image


def gray_max_rgb(inputimagepath,windowname): # 使用最大值进行灰度化
    img = cv2.imread(inputimagepath)#读取图像，返回的是一个装有每一个像素点的bgr值的三维矩阵
    gray_max_rgb_image = img.copy()#复制图像，用于后面保存灰度化后的图像bgr值矩阵
    img_shape = img.shape#返回一位数组（高，宽，3）获得原始图像的长宽以及颜色通道数，一般彩色的颜色通道为3，黑白为1
    for i in range(img_shape[0]):#按行读取图片的像素bgr
        for j in range(img_shape[1]):#对每一行按照列进行每一个像素格子进行读取
            gray_max_rgb_image[i,j] = max(img[i,j][0],img[i,j][1],img[i,j][2])#求灰度值
    # print(gray_max_rgb_image)
    # cv2.imwrite(outimagepath, gray_max_rgb_image)  # 保存当前灰度值处理过后的文件
    return gray_max_rgb_image

def gray_weightmean_rgb(wr,wg,wb,inputimagepath,windowname): #加权平均灰度化
    img = cv2.imread(inputimagepath)
    gray_weightmean_rgb_image = img.copy()
    img_shape = img.shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            gray_weightmean_rgb_image[i,j] = (int(wr*img[i,j][2])+int(wg*img[i,j][1])+int(wb*img[i,j][0]))/3
    return gray_weightmean_rgb_image
def save(value) :            # 将像素数组放入文件中
    print("文件保存中...")
    file = open("pixel.txt", 'w')
    x = 0
    for line in value:
        y = 0
        for v in line :
            file.write('(' + str(x) + ',' + str(y) + '):' + str(v) + ' ')
            y += 1
        x += 1
        file.write('\n')
    print("保存完成")

def read_value(img2) :
    img_array = np.array(img2)
    value = []
    for line in img_array:
        value.append(line)
    save(value)

def test():
    inputimagepath = "image2.jpg"
    windowname = 'gray_cvt'
    # outimagepath = "gray_cvt.jpg"
    img1 = gray_cvt(inputimagepath, windowname)
    img2 = gray_max_rgb(inputimagepath, windowname)
    wr = 0.299
    wg = 0.587
    wb = 0.114
    img3 = gray_weightmean_rgb(wr, wg, wb, inputimagepath, windowname)
    cv2.imshow('gray_cvt', img1)
    cv2.imshow('gray_max_rgb', img2)
    cv2.imshow('gray_weightmean_rgb', img3)
    cv2.waitKey()
    cv2.destroyAllWindows()
    read_value(img1)




if __name__ == '__main__':
    test()