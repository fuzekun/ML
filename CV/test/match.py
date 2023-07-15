"""
    1. 调库实现模板匹配
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def show(name, img) :                        #方便展示图片，方便测试，运行的时候注释掉
    cv.namedWindow(name, cv.WINDOW_NORMAL)    #防止图片太大放不下
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows();

def resize_img(img) :
    # 1. 等比例缩放图片大小
    height, width = img.shape[0:2]
    img = cv.resize(img, (int(width * 1400 / height), int(1400)))
    return img



def test() :
    img = cv.imread('images/ISBN 978-7-01-020356-0.jpg', 0)
    img2 = img.copy()
    template = cv.imread('TrainImage/8.jpg', 0)
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    plt.subplot(4, 2, 1)
    plt.imshow(template, cmap='gray')
    plt.title('Template Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.title('Target Image'), plt.xticks([]), plt.yticks([])

    for i, meth in enumerate(methods):
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv.rectangle(img, top_left, bottom_right, 255, 2)

        f = plt.gcf()  # 获取当前图像
        f.savefig('{}.png'.format(meth))
        f.clear()  # 释放内存
        plt.subplot(4, 2, i+3)
        plt.imshow(img, cmap='gray')
        plt.title('Matching Result by {}'.format(meth)), plt.xticks([]), plt.yticks([])

    plt.show()
if __name__ == '__main__':
    test()