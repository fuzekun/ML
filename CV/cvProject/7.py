import cv2
import cv2 as cv
import numpy as np
import  os
import cv2
import cv2 as cv
import numpy as np
path='c:\\users\\me\\Desktop\\exprienment\\2021\\ISBN 978-7-5073-3874-4.jpg'
#得到字符区域
def get_letter_area():

    global img
    areaT = (len(img) * len(img[0]))
    for i in range(0, len(contours)):
        min__rect = cv.minAreaRect(contours[i])
        if min(min__rect[1])>5:
            area = min__rect[1][0] * min__rect[1][1]
            if (area / areaT >= 0.03 and area / areaT <= 0.07) and min__rect[1][0] + min__rect[1][1] > 900:
                res = cv.drawContours(copyimg, contours[i], -1, (0, 255, 0), 4)
                return i






def get_revolve_angle():              #求旋转角度
    # a=[0,0,0,0]
    # for k in range (0,4):
    #
    #     a[k]=box[k][0]
    #     max_x=a.index(max(a))
    #     min_x=a.index(min(a))
    # if box[max_x][1]>box[min_x][1]:
    #     revolve_angle=90+min_rect[2]
    # else :
    #     revolve_angle = min_rect[2]
    # return revolve_angle
    if min_rect[2]<-45:
        return min_rect[2]+90
    else :
        return min_rect[2]


def resize_img():
    global img
    height,width=img.shape[0:2]###  X表示高度  Y表示宽度
    # if height<1300:
    img = cv.resize(img, (int(width*1400/height), int(1400)), interpolation=cv.INTER_CUBIC)


######################################################################################################################

dir=os.listdir('images')
error_cnt = 0
file=open('error_img.txt','w')
for img_i in range(len(dir)):

    path="images/"+dir[img_i]
    img=cv.imread(path)##读入图片

    resize_img()

    copyimg=img.copy()

    gray=cv.GaussianBlur(img,(3,3),1)
    gray=cv.cvtColor(gray,cv.COLOR_BGR2GRAY)

    ret,binnay_image=cv.threshold(gray,0,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernelX=cv.getStructuringElement(cv.MORPH_RECT,(100,1))
    kernely=cv.getStructuringElement(cv.MORPH_RECT,(4,4))
    binnay=cv.morphologyEx(binnay_image,cv2.MORPH_CLOSE,kernelX)#形态学操作
    binnay=cv.morphologyEx(binnay,cv2.MORPH_OPEN,kernely)
    new=binnay.copy()

    wer,contours,abc=cv.findContours(new,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    #get_letter_area()##可以得到contours[]的序号



    kk=get_letter_area()
    print(img_i, end="")

    if kk==None :
        error_cnt += 1
        file.write(dir[img_i])
        file.write("\n")
        print("错")
        continue



    #res=cv.drawContours(copyimg,contours[1],-1,(0,255,0),4)

    min_rect = cv2.minAreaRect(contours[kk])#返回最小外接矩形的参数（，）   （，）    旋转角
    box =cv.boxPoints(min_rect)
    box=np.int0(box)


    cv.drawContours(copyimg,[box],0,(0,0,255),4)
    shape=binnay_image.shape
    angle=get_revolve_angle()
    M=cv.getRotationMatrix2D(min_rect[0],angle,1.0)
    xuanzhuan=cv.warpAffine(gray,M,(shape[1],shape[0]))#在灰度图旋转
    ret,xuanzhuan=cv.threshold(xuanzhuan,0,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernely=cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    xuanzhuan=cv.dilate(xuanzhuan,kernely)###膨胀
    xuanzhuan=cv.morphologyEx(xuanzhuan,cv2.MORPH_OPEN,kernely)
    xuanzhuan=cv.morphologyEx(xuanzhuan,cv2.MORPH_CLOSE,kernely)
    x=min(min_rect[0])
    y=max(min_rect[0])
    height=min(min_rect[1])                            #  max(int(y-width/2)-5,1)      min(int(y+width/2)+5,width-1)
    width=max(min_rect[1])
    cutimg=xuanzhuan[int(x-height/2):int(x+height/2),max(int(y-width/2)-5,1):min(int(y+width/2)+5,int(shape[1]))]#118行 1759 列
    # kernely = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    #
    # cutimg=cv.dilate(cutimg,kernely)
    print(x,y,height,width)
    print()
    cutimg_height,cutimg_width=cutimg.shape
    cnt=[0]*cutimg_width

    for i in range(0,cutimg_width) :
        for j in range (0,cutimg_height):
            if cutimg[j][i]==255:
                cnt[i]=cnt[i]+1

    mean=[]
    a=[0,cutimg_width-1]
    for i in range (0,cutimg_width-1):
        if (cnt[i]==0 and cnt[i+1]>0 )or (i==0 and cnt[0]>0):
            a[0]=i+1
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
         m = 0
         p = cutimg_height
         height,width=recutimg.shape
         flag=0

         for i in range(0,height):
             for j in range(0,width):
                 if recutimg[i][j]==255:
                     m=i
                     flag=1
                     break
             if flag==1:
                break
         flag=0
         for i in range(1,height):
             for j in range(0, width):
                 if recutimg[cutimg_height-i][int(j)]==255:
                     p=cutimg_height-i
                     flag=1
                     break
             if flag==1:
                 break

         recutimg1 = cutimg[m:p, mean[k][0]:mean[k][1]]
         # cv.namedWindow(str(k), 0)
         # cv.imshow(str(k),recutimg1)

    if len(cutimg) <= 0 or len(cutimg[0]) <= 0 :
        error_cnt += 1
        file.write(dir[img_i])
        file.write("\n")
        print("错")
        continue
    print("对")




    # cv.namedWindow('23',0)
    # cv.namedWindow('gray',0)
    # cv.namedWindow('gray1',0)
    # cv.namedWindow('rot',0)
    # cv.namedWindow('cuting',0)
    # cv.imshow('cuting',cutimg)
    # cv.resizeWindow('23', 640, 240)
    # cv.imshow('gray',copyimg)
    # cv.imshow('23',binnay)
    # cv.imshow('gray1',binnay_image)
    # cv.imshow('rot',xuanzhuan)



    cv.waitKey(200)
    cv.destroyAllWindows()





print("错误" + str(error_cnt))

file.close()











cv.waitKey(0)
cv.destroyAllWindows()