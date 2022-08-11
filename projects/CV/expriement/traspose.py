# import os
# from PIL import Image
# img = Image.open('images/ISBN 978-7-5682-5388-8.jpg')
# img = img.transpose(Image.ROTATE_15)  # 将图片旋转90度
# #img = img.transpose(Image.ROTATE_180)  # 将图片旋转180度
# #img = img.transpose(Image.ROTATE_270)  # 将图片旋转270度
# img.show("img/rotateImg.png")
# #img.save("img/rotateImg.png")

# 导入必要的包
import argparse
import math
import numpy as np
import cv2
import imutils
def test1():
    # 对图像进行旋转
    # USAGE
    # python opencv_rotate.py



    # 构建命令行参数及解析
    # --image 图像路径

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, default="zly1.jpg",
                    help="path to the input image")
    args = vars(ap.parse_args())

    # 从磁盘加载图像并展示
    image = cv2.imread('images/ISBN 978-7-5682-5388-8.jpg')
    image = imutils.resize(image, width=300)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binnay_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 图像灰度二值化

    cv2.imshow("Original", image)

    # 获取图像的维度，并计算中心
    (h, w) = binnay_image.shape
    (cX, cY) = (w // 2, h // 2)

    # 逆时针以图像中心旋转45度
    # - (cX,cY): 旋转的中心点坐标
    # - 45: 旋转的度数，正度数表示逆时针旋转，而负度数表示顺时针旋转。
    # - 1.0：旋转后图像的大小，1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    # OpenCV不会自动为整个旋转图像分配空间，以适应帧。旋转完可能有部分丢失。如果您希望在旋转后使整个图像适合视图，则需要进行优化，使用imutils.rotate_bound.
    pi = 0.12891 * 180 / 3.14
    print(pi)


    M = cv2.getRotationMatrix2D((cX, cY), pi, 1.0)
    rotated = cv2.warpAffine(binnay_image, M, (w, h))

    for line in rotated :
        for x in line:
            print(x, end=" ")

        print()

    cv2.imshow("Rotated by 15 Degrees", rotated)

    # # 逆时针以图像中心旋转-90度图像
    # M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
    # rotated = cv2.warpAffine(image, M, (w, h))
    # cv2.imshow("Rotated by -90 Degrees", rotated)
    #
    # # 以任意点作为中心旋转图像
    # M = cv2.getRotationMatrix2D((10, 10), 45, 1.0)
    # rotated = cv2.warpAffine(image, M, (w, h))
    # cv2.imshow("Rotated by Arbitrary Point", rotated)
    #
    # # 使用imutils.rotata 一行代码实现旋转
    # rotated = imutils.rotate(image, 180)
    # cv2.imshow("Rotated by 180 Degrees", rotated)
    #
    # # 顺时针旋转33度，并保证图像旋转后完整~,确保整个图都在视野范围
    # rotated = imutils.rotate_bound(image, -33)
    # cv2.imshow("Rotated Without Cropping", rotated)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

test1()