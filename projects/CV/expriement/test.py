import cv2


# 绘制直方图

from matplotlib import pyplot as plt

img = cv2.imread('image2.jpg',0)

plt.hist(img.ravel(),256,[0,256])

plt.show()
