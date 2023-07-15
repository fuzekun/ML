import cv2 as cv
import numpy as np
def show(name, img) :
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows();



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
# 2. 根据长宽比例获取数字区域
def get_letter_area(contours, copyimg):
    for i in range(0, len(contours)):
        min__rect = cv.minAreaRect(contours[i])
        if min(min__rect[1]) > 25:
            if float(max(min__rect[1]) / min(min__rect[1])) <= 18 and float(max(min__rect[1]) / min(min__rect[1])) >= 12 and \
                    min__rect[1][0] + min__rect[1][1] > 1300:
                res = cv.drawContours(copyimg, contours[i], -1, (0, 255, 0), 4)
                show('res', res)
                return i

def pretext():


    def get_revolve_angle():              #求旋转角度
        a=[0,0,0,0]
        for k in range (0,4):

            a[k]=box[k][0]
            max_x=a.index(max(a))
            min_x=a.index(min(a))
        if box[max_x][1]>box[min_x][1]:
            revolve_angle=90+min_rect[2]
        else :
            revolve_angle = -(90 + min_rect[2])
        return revolve_angle


    img=cv.imread('images/' + 'ISBN 978-7-5020-8087-7.jpg')

    copyimg=img.copy()


    # 1. 预处理
    gray=cv.GaussianBlur(img,(5,5),1)                           # 高斯滤波
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)                     # 灰度化
    ret,binnay_image=cv.threshold(gray,0,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU) # 二值化
    kernelX=cv.getStructuringElement(cv.MORPH_RECT,(120,1))     # 获取卷积核
    kernely=cv.getStructuringElement(cv.MORPH_RECT,(2,15))
    binnay=cv.morphologyEx(binnay_image,cv.MORPH_CLOSE,kernelX) # 开操作
    binnay=cv.morphologyEx(binnay,cv.MORPH_OPEN,kernely)
    new=binnay.copy()
    show('new', new)

    wer,contours,abc=cv.findContours(new,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) # 获取边缘
    id = get_letter_area(contours, copyimg)##可以得到contours[]的序号

    min_rect = cv.minAreaRect(contours[id])#返回最小外接矩形的参数（，）   （，）    旋转角
    box =cv.boxPoints(min_rect)
    box=np.int0(box)


    cv.drawContours(copyimg,[box],0,(0,0,255),4)
    shape=binnay_image.shape
    M=cv.getRotationMatrix2D(min_rect[0],get_revolve_angle(),1.0)
    xuanzhuan=cv.warpAffine(binnay_image,M,(shape[1],shape[0]))
    x=min(min_rect[0])
    y=max(min_rect[0])
    height=min(min_rect[1])
    width=max(min_rect[1])
    cutimg=xuanzhuan[int(x-height/2)-2:int(x+height/2)+2,int(y-width/2)-5:int(y+width/2)+5]#118行 1759 列
    cutimg_height,cutimg_width=cutimg.shape
    cnt=[0]*cutimg_width

    for i in range(0,cutimg_width):
        for j in range (0,cutimg_height):
            if cutimg[j][i]==255:
                cnt[i]=cnt[i]+1

    mean=[]
    a=[0,cutimg_width-1]
    for i in range (0,cutimg_width-1):
        if (cnt[i]==0 and cnt[i+1]>0 )or (i==0 and cnt[0]>0):
            a[0]=i
        if cnt[i]>0 and cnt[i+1]==0:
            a[1]=i
            if a[1]-a[0]>7:
                mean.append(a)
                a=[0,cutimg_width-1]
        if i==cutimg_width-2>0 and a[0]!=0:
            a[1] = i
            if a[1]-a[0]>7:
                mean.append(a)
                a=[0,cutimg_width-1]
                mean.append(a)

    len_=len(mean)

    for k in range (0,len_):
        recutimg=cutimg[0:cutimg_height,mean[k][0]:mean[k][1]]
        cv.namedWindow(str(k), 0)
        cv.imshow(str(k),recutimg)
























    # cv.namedWindow('23',0)
    # cv.namedWindow('gray',0)
    # cv.namedWindow('gray1',0)
    # cv.namedWindow('rot',0)
    # cv.imshow('cuting',cutimg)
    # #cv.resizeWindow('23', 640, 240)
    # cv.imshow('gray',copyimg)
    # cv.imshow('23',binnay)
    # cv.imshow('gray1',binnay_image)
    # cv.imshow('rot',xuanzhuan)



    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    pretext()