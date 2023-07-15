'''
255代表白色，就是内容

    1. 读取图像，灰度化处理，二值化处理（基于大基算法）
    2. 进行图像的形态学操作，最后得到抠出了前景的背景图片，就只剩下轮廓的图片。
    3. 调整
        1. 需要进行角度的调整，否则行分割难以进行，裂分个也是这样。
        2. 进行bfs进行滤波
    3. 将图像进行上下切割得到ISBN号码，
        3.1 计算水平方向像素点的个数，进行边界分割，如何判断第几行是所需要的(0,1,2)
        3.2 根据像素点的个数进行分割
        3.2 计算竖直方向的个数，分割出单个字符。
    4. 进行数字的识别
    方法一 : 4.1 识别出图像的边缘，(使用canny算法)
            4.2 使用图形学的基元匹配

'''
import math

import cv2 as cv
import numpy as np
import imutils
def show(name, img) :                        #方便展示图片，方便测试，运行的时候注释掉
    cv.namedWindow(name, cv.WINDOW_NORMAL)    #防止图片太大放不下
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows();

color = 255                            #确定背景图片是0还是1
def transpos(image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 逆时针以图像中心旋转45度
    # - (cX,cY): 旋转的中心点坐标
    # - 45: 旋转的度数，正度数表示逆时针旋转，而负度数表示顺时针旋转。
    # - 1.0：旋转后图像的大小，1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    # OpenCV不会自动为整个旋转图像分配空间，以适应帧。旋转完可能有部分丢失。如果您希望在旋转后使整个图像适合视图，则需要进行优化，使用imutils.rotate_bound.
    M = cv.getRotationMatrix2D((cX, cY), 5, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    show("Rotated by 5 Degrees", rotated)
    return rotated

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
def preHandle(imgPath) :                #图像的预处理
    img = cv.imread(imgPath)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 灰度化
    # 二值化
    ret1, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU ) #使用大基算法，将图像进行二值化
    # thresh = mythresh(gray)                  #注意结合大基算法
    # save(thresh)

    ret = addBounder(thresh)                   #增加边框
    show('preImage', ret)
    return ret

# 将存储区域特别小的改成0
def changeTo0(img, tmp) :
    for (x, y) in tmp:
        img[x, y] = 0

# 找联通区域，返回联通区域的最低值
recs = []                                       #保存联通区域的上下左右边界
def bfs(grid, x, y, vis, tmpa) :                #tmpa保存内容
    dir = [(-1, 0), (1, 0), (0, -1), (0, 1)    # 上下左右
        , (-1, -1), (1, -1), (-1, 1), (1, 1)]  # 左上下，右上下
    (m, n) = grid.shape
    top = m
    bottom = 0
    left = n
    right = 0
    que = [(x, y)]
    tmpa.append((x, y))
    while len(que) > 0 :
        (x, y) = que.pop(0)                    #取出队列头
        for (dirx, diry) in dir :
            nx = x + dirx
            ny = y + diry
            if nx >= 0 and nx < m and ny >= 0 and ny < n and vis[nx][ny] == 0 and grid[nx][ny] == 255:
                bottom = max(bottom, nx)
                top = min(top, nx)
                left = min(left, ny)
                right = max(right, ny)
                que.append((nx, ny))
                tmpa.append((nx, ny))
                vis[nx][ny] = 1
    return (top, bottom, left, right)

def bfs2(grid, x, y, vis, cnts) :                       #cnts保存数量
    dir = [(-1, 0), (1, 0), (0, -1), (0, 1)  # 上下左右
        , (-1, -1), (1, -1), (-1, 1), (1, 1)]  # 左上下，右上下
    (m, n) = grid.shape
    top = m
    bottom = 0
    left = n
    right = 0
    que = [(x, y)]
    cnt = 0
    while len(que) > 0 :
        (x, y) = que.pop(0)                    #取出队列头
        for (dirx, diry) in dir :
            nx = x + dirx
            ny = y + diry
            if nx >= 0 and nx < m and ny >= 0 and ny < n and vis[nx][ny] == 0 and grid[nx][ny] == 255:
                bottom = max(bottom, nx)
                top = min(top, nx)
                left = min(left, ny)
                right = max(right, ny)
                que.append((nx, ny))
                cnt += 1
                vis[nx][ny] = 1
    cnts.append(cnt)
    return (top, bottom, left, right)


"""
     :param image: 预处理之后的图像
     :param thresh: 去噪声的阈值
     :return: 返回旋转的角度 
"""
def findConnection(image, thresh) :
    (maxx, maxy) = image.shape
    vis = np.zeros((maxx + 5, maxy + 5))
    # 去除边框
    tmpb = []
    bfs(image, 0, 0, vis, tmpb)
    changeTo0(img, tmpb)

    for i in range(maxx):                               #找到前两个边界值
        for j in range(maxy):
            if image[i][j] == 255 and vis[i][j] == 0:
                tmpa = []
                rec = bfs2(image, i, j, vis, tmpa)          # 连通区域的边界
                # print("lena = " + str(len(tmpa)))
                if tmpa[0] > thresh:                        # 如果不是噪声点
                    recs.append(rec)                        # 保存每个字符的边界
                    if len(recs) >= 2 : break               # 找到前两个字符就行
        if len(recs) >= 2 : break
    tanx = (recs[1][0] - recs[0][0]) / (recs[1][2] - recs[0][2])
    degree = np.arctan(tanx) * 180 / math.pi
    print("角度 = " + str(degree))
    return degree


def getBounder(image, thresh) :                     #由findC..变形而来，找旋转后字符的上下边界
    (maxx, maxy) = image.shape
    vis = np.zeros((maxx + 5, maxy + 5))
    bottom = 0
    top = maxx
                                                    #可能需要再一次进行预处理
    for i in range(maxx):
        flag = 0                                    # 判断是否是二维码
        for j in range(maxy):
            if image[i][j] == 255 and vis[i][j] == 0:
                tmpa = []
                rec = bfs(image, i, j, vis, tmpa)          # 联通区域的边界
                # print("n_lena = " + str(len(tmpa)))
                if len(tmpa) > thresh:                     # 如果不是噪声点
                    if bottom != 0 and rec[1] > 1.5 * bottom :           # 如果是二维码
                        print("扫描到二维码啦!")
                        flag = 1
                        break
                    top = min(top, rec[0])
                    bottom = max(bottom, rec[1])
                else :                                     # 如果是噪声点，去噪
                    changeTo0(img, tmpa)
        if flag == 1:
            break
    return (top, bottom)
def draw(image) :
    show('changeTo0', image)
    recs.sort(key=lambda x:(x[0], x[2]))         #根据列进行排序,之后根据列
    print(len(recs))
    for i in range(len(recs)):
        rec = recs[i]
        # print("Rec")
        # for i in rec:
        #     print(i, end = " ")
        # print()
        img = image[rec[0]: rec[1], rec[2]: rec[3]]
        show('char_' + str(i), img)

def transpos(image, degree):                                #将图像进行旋转
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv.getRotationMatrix2D((cX, cY), degree, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    show("Rotated by x Degrees", rotated)
    return rotated


def cutLine(image, rec):
    img  = image[rec[0]:rec[1], :]
    show('line', img)
    return img

def bfs3(grid, x, y) :
    dir = [(-1, 0), (1, 0), (0, -1), (0, 1)    # 上下左右
        , (-1, -1), (1, -1), (-1, 1), (1, 1)]  # 左上下，右上下
    m = len(grid)
    n = len(grid[0])
    top = m
    bottom = 0
    left = n
    right = 0
    que = [(x, y)]
    vis = np.zeros((m, n))
    while len(que) > 0 :
        (x, y) = que.pop(0)                    #取出对头
        for (dirx, diry) in dir :
            nx = x + dirx
            ny = y + diry
            if nx >= 0 and nx < m and ny >= 0 and ny < n and vis[nx][ny] == 0 and grid[nx][ny] == 255:
                que.append((nx, ny))
                top = min(top, nx)
                bottom = max(bottom, nx)
                left = min(left, ny)
                right = max(right, ny)
                vis[nx][ny] = 1
    return [top, bottom, left, right]

def fstart(img) :
    (x, y) = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] == 255 :
                return (i, j)

def solveWImg(img) :
    (x, y) = fstart(img)
    boder = bfs3(img, x, y)
    img = img[boder[0]:boder[1], boder[2]:boder[3]]
    return img

def columnCut(img):                         #图片进行列分割
    (x, y) = img.shape
    cnt = np.zeros(y)
    for j in range(y):
        for i in range(x):
            if img[i, j] == 255:              #白色像素255，代表内容
                cnt[j] += 1
    # plt.plot(range(x), cnt)
    # plt.show()
    start=[]                                 #开始索引数组
    end=[]                                   #结束索引数组
    if cnt[0] != 0 : start.append(0)         #开始边界
    for index in range(1, y - 1):
        #本列大于0，上一列等于0，即开始
        if cnt[index] != 0 and cnt[index - 1] == 0:
            start.append(index)
        #本列大于0，下一列等0，即结束
        elif cnt[index + 1] == 0 and cnt[index] != 0:
             end.append(index)
    if cnt[y - 1] != 0 : end.append(x - 1)      #结束边界
    imgs = []
    print("col.len = "  + str(min(len(start), len(end))))
    for i in range(min(len(start), len(end))):
        print("c_start = " + str(start[i]))       #输出防止start > end
        print("c_end = " + str(end[i]))
        if start[i] >= end[i] :
            continue
        imgi = img[:, start[i]:end[i]]
        imgs.append(imgi)
        imgi = solveWImg(imgi)
        if len(img) *  len(imgi[0]) > 50:        #有可能截取出来噪声点
            show("c_" + str(i), imgi)
        cv.imwrite("numbers/" + str(i) + ".jpg", imgi)
    return imgs


if __name__ == '__main__':
    img = preHandle("images/ISBN 978-7-101-13179-6.JPG")   #ISBN 978-7-300-23418-2.jpg
    degree = findConnection(img, 50)
    rotated = transpos(img, degree)                 #将图像进行旋转
    rec = getBounder(rotated, 50)                   #获取旋转图像的左右边界
    print("top = " + str(rec[0]) + ' bottom = ' + str(rec[1]))
    img = cutLine(rotated, rec)
    img = columnCut(img)
    # draw(img)