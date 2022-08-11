
import numpy as np
import cv2 as cv

image = cv.imread("images/ISBN 978-7-01-020356-0.jpg", 0)  #转化为单通道的灰度图
img_Guassian = cv.GaussianBlur(image,(5,5),0)

ret1, thresh = cv.threshold(img_Guassian, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU ) #使用大基算法，将图像进行二值化



kernel = np.ones((3, 3), np.uint8) #生成一个一行一列的整形结构元
for i in range(len(kernel)):
    for j in range(len(kernel[0])) :
        if j == 1:
            continue
        kernel[i][j] = 0


imageopen = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2) #形态学转换,腐蚀后膨胀(开操作)
cv.imshow("imageopen", imageopen)
bg = cv.dilate(imageopen, kernel, iterations=3) # 使用特定的结构元进行膨胀操作。


distTransform = cv.distanceTransform(imageopen, cv.DIST_L2, 5) # 前景分割
ret2, fore = cv.threshold(distTransform, 0.4*distTransform.max(), 255, 0)

fore = np.uint8(fore)
un = cv.subtract(bg, fore)   # 图像剑法，背景减去前景，剩下的就是所需的图片
cv.imshow('gass', img_Guassian)

cv.imshow("imageGray", image)
cv.imshow("bg", bg)
cv.imshow("fore", fore)
cv.imshow("un", un)

cv.waitKey()
cv.destroyAllWindows()

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
#
#
# # 1. 进行前景分割
# image = cv.imread("image2.jpg")
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# imagergb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#
# ret1, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# kernel = np.ones((1, 1), np.uint8)
#
# opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
# sure_bg = cv.dilate(opening, kernel, iterations=3)
#
# dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
# ret2, sure_fg = cv.threshold(dist_transform, 0.005*dist_transform.max(), 255, 0)
# sure_fg = np.uint8(sure_fg)
#
# unknown = cv.subtract(sure_bg, sure_fg)
#
# ret3, markers = cv.connectedComponents(sure_fg)
# img = cv.watershed(image, markers)
#
# plt.subplot(121)
# plt.imshow(imagergb)
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(img)
# plt.axis('off')
#
# plt.show()
# cv.waitKey()
# cv.destroyAllWindows()



# import numpy as np
# import cv2 as cv
#
# image = cv.imread("image2.jpg", 0)
# ret1, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
#
# kernel = np.ones((2, 2), np.uint8)
#
# imageopen = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
#
# distTransform = cv.distanceTransform(imageopen, cv.DIST_L2, 5)
# ret2, fore = cv.threshold(distTransform, 0.4*distTransform.max(), 255, 0)
#
# cv.imshow("imageGray", image)
# cv.imshow("imageOpen", imageopen)
# cv.imshow("distTransform", distTransform)
# cv.imshow("fore", fore)
#
# cv.waitKey()
# cv.destroyAllWindows()


import cv2
import numpy as np

# def waterShed(sourceDir):
# 	# 读取图片
# 	img = cv2.imread(sourceDir)
# 	# 原图灰度处理,输出单通道图片
# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	# 二值化处理Otsu算法
# 	reval_O, dst_Otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# 	# 二值化处理Triangle算法
# 	reval_T, dst_Tri = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
# 	# 滑动窗口尺寸
# 	kernel = np.ones((3, 3), np.uint8)
# 	# 形态学处理:开处理,膨胀边缘
# 	opening = cv2.morphologyEx(dst_Tri, cv2.MORPH_OPEN, kernel, iterations=2)
# 	# 膨胀处理背景区域
# 	dilate_bg = cv2.dilate(opening, kernel, iterations=3)
# 	# 计算开处理图像到邻域非零像素距离
# 	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# 	# 正则处理
# 	norm = cv2.normalize(dist_transform, 0, 255, cv2.NORM_MINMAX)
# 	# 阈值处理距离图像,获取图像前景图
# 	retval_D, dst_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
# 	# 前景图格式转换
# 	dst_fg = np.uint8(dst_fg)
# 	# 未知区域计算:背景减去前景
# 	unknown = cv2.subtract(dilate_bg, dst_fg)
# 	cv2.imshow("Difference value", unknown)
# 	cv2.imwrite('./images/saved/unknown_reginon.png', unknown)
# 	# 处理连接区域
# 	retval_C, marks = cv2.connectedComponents(dst_fg)
# 	cv2.imshow('Connect marks', marks)
# 	cv2.imwrite('./images/saved/connect_marks.png', marks)
# 	# 处理掩模
# 	marks = marks + 1
# 	marks[unknown==255] = 0
# 	cv2.imshow("marks undown", marks)
# 	# 分水岭算法分割
# 	marks = cv2.watershed(img, marks)
# 	# 绘制分割线
# 	img[marks == -1] = [255, 0, 255]
# 	cv2.imshow("Watershed", img)
# 	cv2.imwrite('./images/saved/watershed.png', img)
# 	cv2.waitKey(0)
# sourceDir = "image2.jpg"
# waterShed(sourceDir)
#

# def cutImage(sourceDir):
# 	# 读取图片
# 	img = cv2.imread(sourceDir)
# 	# 灰度化
# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	# 高斯模糊处理:去噪(效果最好)
# 	blur = cv2.GaussianBlur(gray, (9, 9), 0)
# 	# Sobel计算XY方向梯度
# 	gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
# 	gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)
# 	# 计算梯度差
# 	gradient = cv2.subtract(gradX, gradY)
# 	# 绝对值
# 	gradient = cv2.convertScaleAbs(gradient)
# 	# 高斯模糊处理:去噪(效果最好)
# 	blured = cv2.GaussianBlur(gradient, (9, 9), 0)
# 	# 二值化
# 	_ , dst = cv2.threshold(blured, 90, 255, cv2.THRESH_BINARY)
# 	# 滑动窗口
# 	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (107, 76))
# 	# 形态学处理:形态闭处理(腐蚀)
# 	closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
# 	# 腐蚀与膨胀迭代
# 	closed = cv2.erode(closed, None, iterations=4)
# 	closed = cv2.dilate(closed, None, iterations=4)
# 	# 获取轮廓
# 	_, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 	c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
# 	rect = cv2.minAreaRect(c)
# 	box = np.int0(cv2.boxPoints(rect))
# 	draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
# 	cv2.imshow("Box", draw_img)
# 	cv2.imwrite('./images/saved/monkey.png', draw_img)
# 	cv2.waitKey(0)
# sourceDir = "image2.jpg"
# cutImage(sourceDir)
# def grab_cut(sourceDir):
# 	# 读取图片
# 	img = cv2.imread(sourceDir)
# 	# 图片宽度
# 	img_x = img.shape[1]
# 	# 图片高度
# 	img_y = img.shape[0]
# 	# 分割的矩形区域
# 	rect = (96,1, 359, 358)
# 	# 背景模式,必须为1行,13x5列
# 	bgModel = np.zeros((1, 65), np.float64)
# 	# 前景模式,必须为1行,13x5列
# 	fgModel = np.zeros((1, 65), np.float64)
# 	# 图像掩模,取值有0,1,2,3
# 	mask = np.zeros(img.shape[:2], np.uint8)
# 	# grabCut处理,GC_INIT_WITH_RECT模式
# 	cv2.grabCut(img, mask, rect, bgModel, fgModel, 4, cv2.GC_INIT_WITH_RECT)
# 	# grabCut处理,GC_INIT_WITH_MASK模式
# 	# cv2.grabCut(img, mask, rect, bgModel, fgModel, 4, cv2.GC_INIT_WITH_MASK)
# 	# 将背景0,2设成0,其余设成1
# 	mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
# 	# 重新计算图像着色,对应元素相乘
# 	img = img*mask2[:, :, np.newaxis]
# 	cv2.imshow("Result", img)
# 	cv2.waitKey(0)
# sourceDir = "image2.jpg"
# grab_cut(sourceDir)
