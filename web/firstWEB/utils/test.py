import os
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 1.测试img_write和img_read相对路径的写法
    img = cv.imread("1.jpg")
    cv.imshow("test", img)
    cv.waitKey(0)
    cv.imwrite("../static/fire/1619509588876.jpg", img)