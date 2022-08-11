import numpy as np
import cv2
from PIL import  Image


def read() :
    img = cv2.imread('image2.jpg')            # 读取图像,并灰度化
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)  # 给窗口命名
    cv2.imshow('image', img)                    # 在窗口中显示img
    cv2.waitKey(0)                              # 等待按键
    cv2.destroyAllWindows()                     # 关闭窗口



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


def test() :
    img = Image.open('image2.jpg')
    img = img.convert('L') # 灰度化
    cols, rows = img.size

    print('colr = ' + str(cols))
    print('rows = ' + str(rows))
    img_array = np.array(img)
    value = []
    for line in img_array :
        value.append(line)
    save(value)


read()