# 看canny返回的图像是否没有噪声

#Canny边缘提取
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pylab

def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0) # 高斯平滑
    cv.namedWindow('blurred', cv.WINDOW_NORMAL)  # 设置为WINDOW_NORMAL可以任意缩放
    cv.imshow("blurred", blurred)
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY) # 灰度转换
    # xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0) #x方向梯度
    # ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1) #y方向梯度
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    #其中第9行代码可以用6、7、8行代码代替！两种方法效果一样。
    edge_output = cv.Canny(gray, 50, 150)
    cv.namedWindow('Canny Edge', cv.WINDOW_NORMAL)  # 设置为WINDOW_NORMAL可以任意缩放
    cv.imshow("Canny Edge", edge_output)
    dst = cv.bitwise_and(image, image, mask= edge_output)
    cv.namedWindow('Color Edge', cv.WINDOW_NORMAL)
    cv.imshow("Color Edge", dst)

def test1() :               #库 的高斯滤波 与 边缘检测
    src = cv.imread('images/ISBN 978-7-01-020356-0.jpg')
    cv.namedWindow('input_image', cv.WINDOW_NORMAL)  # 设置为WINDOW_NORMAL可以任意缩放
    cv.imshow('input_image', src)
    edge_demo(src)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    test1()