import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys



"""
第一个实验，采用三种方法进行灰度化处理：
1. 库函数
2. RGB最大值
3. RGB均值

直方图放在第二个实验中
"""

def gray_cvt(inputimagepath): #使用cv默认的方法灰度化
    img = cv.imread(inputimagepath)
    gray_cvt_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 灰度化
    # cv.imwrite(outimagepath, gray_cvt_image)  # 保存当前灰度值处理过后的文件
    return gray_cvt_image


def gray_max_rgb(inputimagepath): # 使用最大值进行灰度化
    img = cv.imread(inputimagepath)#读取图像，返回的是一个装有每一个像素点的bgr值的三维矩阵
    gray_max_rgb_image = img.copy()#复制图像，用于后面保存灰度化后的图像bgr值矩阵
    img_shape = img.shape#返回一位数组（高，宽，3）获得原始图像的长宽以及颜色通道数，一般彩色的颜色通道为3，黑白为1
    for i in range(img_shape[0]):#按行读取图片的像素bgr
        for j in range(img_shape[1]):#对每一行按照列进行每一个像素格子进行读取
            gray_max_rgb_image[i,j] = max(img[i,j][0],img[i,j][1],img[i,j][2])#求灰度值
    # print(gray_max_rgb_image)
    # cv.imwrite(outimagepath, gray_max_rgb_image)  # 保存当前灰度值处理过后的文件
    return gray_max_rgb_image

def gray_weightmean_rgb(wr,wg,wb,inputimagepath): #加权平均灰度化
    img = cv.imread(inputimagepath)
    gray_weightmean_rgb_image = img.copy()
    img_shape = img.shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            gray_weightmean_rgb_image[i,j] = (int(wr*img[i,j][2])+int(wg*img[i,j][1])+int(wb*img[i,j][0]))/3
    return gray_weightmean_rgb_image

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
    print("保存完成")

def read_value(img2) :
    img_array = np.array(img2)
    value = []
    for line in img_array:
        value.append(line)
    save(value)

def first():                                #第一个实验主函数
    inputimagepath = "image2.jpg"
    windowname = 'gray_cvt'
    # outimagepath = "gray_cvt.jpg"
    img1 = gray_cvt(inputimagepath)
    img2 = gray_max_rgb(inputimagepath)
    wr = 0.299
    wg = 0.587
    wb = 0.114
    img3 = gray_weightmean_rgb(wr, wg, wb, inputimagepath)
    cv.imshow('gray_cvt', img1)
    cv.imshow('gray_max_rgb', img2)
    cv.imshow('gray_weightmean_rgb', img3)
    cv.waitKey()
    cv.destroyAllWindows()
    read_value(img1)


"""
    第二个实验：
    绘制灰度直方图 + 大基算法进行阈值分割
    1. 直方图手动统计使用绘图函数绘制
    2. 阈值采用库函数和大基算法进行分割
"""
def changeToArray() :               #转化为数组
    img = gray_cvt('image2.jpg')
    image = np.array(img)
    value = []
    for line in image:
        value.append(line)
    return value



def sdraw() : #绘制灰度直方图
    value = changeToArray()
    # 1.统计每一个灰度值的
    val = []
    num = np.zeros(256)
    for line in value:
        for v in line:
            val.append(v)
            num[v] += 1
    # 自动统计value中的每一个值有多少
    plt.plot(range(256), num)
    plt.hist(val, bins = 256, rwidth=2, color='yellow')
    plt.savefig('pixel.png')  #保存图像
    plt.show()



def rgb2gray(rgb):
    # 将rgb转灰度
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def otsu(gray) : #使用大基算法得到最佳的阈值分割点

    #    return 0.2989 * image[:,:,0] + 0.5870*image[:,:,1] + 0.1140*image[:,:,2]
    h, x = np.histogram(gray.ravel(), bins=256)

    plt.imshow(gray, cmap=plt.get_cmap('gray'))  # 显示灰度图
    plt.show()

    # 整体均值
    meanall = np.sum(np.dot(h, np.array([n for n in range(256)])))
    meanall = meanall / np.sum(np.array([n for n in range(256)]))
    maxscore = 0;
    gi = []
    for i in range(1, 255):
        # 分割图像的均值只需要将直方图的高度乘以x坐标求和，再除以x坐标之和
        mean1 = np.sum(np.dot(h[0:i], np.array([n for n in range(i)])))
        mean1 = mean1 / np.sum(h[0:i])
        mean2 = np.sum(np.dot(h[i:256], np.array([n for n in range(i, 256)])))
        mean2 = mean2 / np.sum(h[i:256])
        # 公式计算
        score = sum(h[0:i]) * ((meanall - mean1) ** 2) + sum(h[i:256]) * ((meanall - mean2) ** 2)

        #    用于绘图
        gi.append(score)
        if maxscore < score:  # 记录最大值
            maxscore = score
            threshold = i

    # print("max value = %d, th = %d" % (maxscore, threshold))

    # 用于绘图
    gi.append(min(gi))
    gi.insert(0, min(gi))
    plot1 = plt.figure()
    # 绘制直方图
    plt.bar(np.array([n for n in range(256)]), h)
    # 绘制分割点
    plt.axvline(threshold, color='r')
    # 绘制类间方差遍历过程示意图
    plt.scatter([n for n in range(256)], (gi - min(gi)) / (max(gi) - min(gi)) * max(h))
    plt.show()
    return threshold


'''
1. 先将图片灰度化处理
2. 将处理后的图片进行选择阈值之后进行固定阈值全局分割，前景为0，背景为255
3. 分割之后
'''

def doThreshold(img, thresh) : #实际实现分割的函数
    ret = np.zeros((len(img), len(img[0])), np.uint8)
    for i in range(len(img)) :
        for j in range(len(img[0])) :
            if img[i][j] > thresh :
                ret[i][j] = 255
            else :
                ret[i][j] = 0
    return ret
def myThreshold() :
    # 1.灰度化处理
    image = gray_cvt('image2.jpg')
    # 2. 使用大基算法得到阈值
    thresh = otsu(image)

    # 3. 使用阈值进行固定阈值分割图像
    img = doThreshold(image, thresh)
    cv.imshow("besthresh", img)

    # 4. 使用肉眼观察的进行对比
    img = doThreshold(image, 100)
    cv.imshow("thresh", img)

    cv.waitKey(0)
    cv.destroyAllWindows()


def second() :          #第二个实验的主函数
    sdraw()              #绘制二分图像
    myThreshold()       #进行灰度化的处理

"""
第三个实验:
1. 采用库函数的Canny算法进行边缘提取
2. 采用手写的Canny算法进行边缘提取
"""
#Canny边缘提取
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
    cv.imshow("Color Edge", dst)

def test1() :               #库 的高斯滤波 与 边缘检测
    src = cv.imread('image2.jpg')
    cv.namedWindow('input_image', cv.WINDOW_NORMAL)  # 设置为WINDOW_NORMAL可以任意缩放
    cv.imshow('input_image', src)
    edge_demo(src)
    cv.waitKey(0)
    cv.destroyAllWindows()

'''
    默认采用same模式，可以采用full模式
'''
def convolve(img,fil,mode = 'same'):                #对图像的每一个通道进行卷积

    if mode == 'fill':
        h = fil.shape[0] // 2
        w = fil.shape[1] // 2
        img = np.pad(img, ((h, h), (w, w),(0, 0)), 'constant')
    conv_b = _convolve(img[:,:,0],fil)              #然后去进行卷积操作
    conv_g = _convolve(img[:,:,1],fil)
    conv_r = _convolve(img[:,:,2],fil)

    dstack = np.dstack([conv_b,conv_g,conv_r])      #将卷积后的三个通道合并
    return dstack                                   #返回卷积后的结果
def _convolve(img,fil):                             #卷积

    fil_heigh = fil.shape[0]                        #获取卷积核(滤波)的高度
    fil_width = fil.shape[1]                        #获取卷积核(滤波)的宽度

    conv_heigh = img.shape[0] - fil.shape[0] + 1    #确定卷积结果的大小
    conv_width = img.shape[1] - fil.shape[1] + 1

    conv = np.zeros((conv_heigh,conv_width),dtype = 'uint8')

    for i in range(conv_heigh):
        for j in range(conv_width):                 #逐点相乘并求和得到每一个点
            conv[i][j] = wise_element_sum(img[i:i + fil_heigh,j:j + fil_width ],fil)
    return conv

def wise_element_sum(img,fil):
    res = (img * fil).sum()
    if(res < 0):
        res = 0
    elif res > 255:
        res  = 255
    return res

def blur() : #平滑

    img = plt.imread("image2.jpg")  # 在这里读取图片
    plt.imshow(img)  # 显示读取的图片
    pylab.show()

    # 高斯平滑卷积核
    fil = np.array([[0.05, 0.1, 0.05],
                    [0.1, 0.4, 0.1],
                    [0.05, 0.1, 0.05]], dtype=np.float)


    res = convolve(img, fil, 'fill')
    # print("img shape :" + str(img.shape))
    # plt.imshow(res)  # 显示卷积后的图片
    # print("res shape :" + str(res.shape))
    plt.imsave("res.jpg", res)
    # pylab.show()
    return res




def Canny(img):
    # Gray scale
    def BGR2GRAY(img):
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        # Gray scale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        return out

    # Gaussian filter for grayscale
    def gaussian_filter(img, K_size=3, sigma=1.4):

        if len(img.shape) == 3:
            H, W, C = img.shape
            gray = False
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
            gray = True

        ## Zero padding
        pad = K_size // 2
        out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

        ## prepare Kernel
        K = np.zeros((K_size, K_size), dtype=np.float)
        for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                K[y + pad, x + pad] = np.exp(- (x ** 2 + y ** 2) / (2 * sigma * sigma))
        # K /= (sigma * np.sqrt(2 * np.pi))
        K /= (2 * np.pi * sigma * sigma)
        K /= K.sum()

        tmp = out.copy()

        # filtering
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

        out = np.clip(out, 0, 255)
        out = out[pad: pad + H, pad: pad + W]
        out = out.astype(np.uint8)

        if gray:
            out = out[..., 0]

        return out

    # sobel filter
    def sobel_filter(img, K_size=3):
        if len(img.shape) == 3:
            H, W, C = img.shape
        else:
            H, W = img.shape

        # Zero padding
        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        ## Sobel vertical
        Kv = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
        ## Sobel horizontal
        Kh = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

        # filtering
        for y in range(H):
            for x in range(W):
                out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
                out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))

        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad: pad + H, pad: pad + W]
        out_v = out_v.astype(np.uint8)
        out_h = out_h[pad: pad + H, pad: pad + W]
        out_h = out_h.astype(np.uint8)

        return out_v, out_h

    # get edge strength and edge angle
    def get_edge_angle(fx, fy):
        # get edge strength
        edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
        edge = np.clip(edge, 0, 255)

        # make sure the denominator is not 0
        fx = np.maximum(fx, 1e-10)
        # fx[np.abs(fx) <= 1e-5] = 1e-5

        # get edge angle
        angle = np.arctan(fy / fx)

        return edge, angle

    # 将角度量化为0°、45°、90°、135°
    def angle_quantization(angle):
        angle = angle / np.pi * 180
        angle[angle < -22.5] = 180 + angle[angle < -22.5]
        _angle = np.zeros_like(angle, dtype=np.uint8)
        _angle[np.where(angle <= 22.5)] = 0
        _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
        _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
        _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

        return _angle

    def non_maximum_suppression(angle, edge):
        H, W = angle.shape
        _edge = edge.copy()

        for y in range(H):
            for x in range(W):
                if angle[y, x] == 0:
                    dx1, dy1, dx2, dy2 = -1, 0, 1, 0
                elif angle[y, x] == 45:
                    dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                elif angle[y, x] == 90:
                    dx1, dy1, dx2, dy2 = 0, -1, 0, 1
                elif angle[y, x] == 135:
                    dx1, dy1, dx2, dy2 = -1, -1, 1, 1
                # 边界处理
                if x == 0:
                    dx1 = max(dx1, 0)
                    dx2 = max(dx2, 0)
                if x == W - 1:
                    dx1 = min(dx1, 0)
                    dx2 = min(dx2, 0)
                if y == 0:
                    dy1 = max(dy1, 0)
                    dy2 = max(dy2, 0)
                if y == H - 1:
                    dy1 = min(dy1, 0)
                    dy2 = min(dy2, 0)
                # 如果不是最大值，则将这个位置像素值置为0
                if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                    _edge[y, x] = 0

        return _edge

    # 滞后阈值处理二值化图像
    # > HT 的设为255，< LT 的设置0，介于它们两个中间的值，使用8邻域判断法
    def hysterisis(edge, HT=100, LT=30):
        H, W = edge.shape

        # Histeresis threshold
        edge[edge >= HT] = 255
        edge[edge <= LT] = 0

        _edge = np.zeros((H + 2, W + 2), dtype=np.float32)
        _edge[1: H + 1, 1: W + 1] = edge

        ## 8 - Nearest neighbor
        nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

        for y in range(1, H + 2):
            for x in range(1, W + 2):
                if _edge[y, x] < LT or _edge[y, x] > HT:
                    continue
                if np.max(_edge[y - 1:y + 2, x - 1:x + 2] * nn) >= HT:
                    _edge[y, x] = 255
                else:
                    _edge[y, x] = 0

        edge = _edge[1:H + 1, 1:W + 1]

        return edge

    # grayscale
    gray = BGR2GRAY(img)

    # gaussian filtering
    gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)

    # sobel filtering
    fy, fx = sobel_filter(gaussian, K_size=3)

    # get edge strength, angle
    edge, angle = get_edge_angle(fx, fy)

    # angle quantization
    angle = angle_quantization(angle)

    # non maximum suppression
    edge = non_maximum_suppression(angle, edge)

    # hysterisis threshold
    out = hysterisis(edge, 100, 50)

    return out


def test2():
    img = cv.imread("image2.jpg").astype(np.float32)
    edge = Canny(img)

    out = edge.astype(np.uint8)
    # Save result

    cv.imwrite("out.jpg", out)
    cv.imshow("result", out)
    cv.waitKey(0)
    cv.destroyAllWindows()

def third() :                   # 实验三主函数
    test1()


"""
    实验四
    1. 分别使用bfs和dfs两种方式找到第一个连通区域
    2. 使用bfs找到每一个8连通区域
    3. 手算每一个连通区域的重心和面积
    4. 手算每一个连通区域的最小平行轴矩形
    5. 手绘出最小平行轴外接矩形
    6. 得到最小平行轴外界矩形的面积
"""

def mythresh(image, thresh = 135) :
    img = []
    for line in image:
        tmp = [255 if s > thresh else 0 for s in line]
        img.append(tmp)
    img = np.array(img, np.uint8)
    return img


ans = []
anss = []
dir = [(-1, 0), (1, 0), (0, -1), (0, 1)  # 上下左右
    , (-1, -1), (1, -1), (-1, 1), (1, 1)]  # 左上下，右上下
def dfs(grid, x, y, maxx, maxy, vis) :
    vis[x][y] = 1
    ans.append((x, y))
    if len(ans) > 1000 :           # 防止像素太多爆栈
        return
    # print("(" + str(x) + ',' + str(y) + ") ")
    for (dirx, diry) in dir:
        nx = dirx + x
        ny = diry + y
        if nx >= 0 and nx < maxx and ny >= 0 and ny < maxy and vis[nx][ny] == 0 and grid[nx][ny] == 0:
            dfs(grid, nx, ny, maxx, maxy, vis)



def bfs(grid, x, y, vis, tmpa) :
    m = len(grid)
    n = len(grid[0])
    que = [(x, y)]
    tmpa.append((x, y))
    while len(que) > 0 :
        (x, y) = que.pop(0)                    #取出对头
        for (dirx, diry) in dir :
            nx = x + dirx
            ny = y + diry
            if nx >= 0 and nx < m and ny >= 0 and ny < n and vis[nx][ny] == 0 and grid[nx][ny] == 0:
                que.append((nx, ny))
                tmpa.append((nx, ny))
                vis[nx][ny] = 1

def findConnection(image) :             # 按照行优先，找到第一个像素为255(1)的联通区域。
    #前景是0，找到第一个为0的就是像素
    maxx = len(image)
    maxy = len(image[0])
    vis = np.zeros((maxx + 5, maxy + 5))
    for i in range(maxx):
        for j in range(maxy):
            if image[i][j] == 0 and vis[i][j] == 0:
                tmpa = []
                # dfs(image, i, j, maxx, maxy, vis)
                bfs(image, i , j, vis, tmpa)
                if len(tmpa) > 200 :             # 去除噪声
                    anss.append(tmpa)

def draw(grid, hegith, widht) :                  #绘制图像
    image = np.zeros((hegith, widht), np.uint8)
    for (x, y)  in grid:
        image[x][y] =  255
    return image

def getAear(grid) :                         #获取每一个区域到面积
    return len(grid)

def getFoucs(image):                        #获取重心
    area = len(image)                       #面积
    sum_x = 0
    sum_y = 0
    for (i, j) in image :
        sum_x += i
        sum_y += j
    return (sum_x / area, sum_y / area)

def getRec(grid) :   #得到最小的平行外界矩形
    x1 = sys.maxsize        #左上角和右下角
    y1 = sys.maxsize
    x2 = 0
    y2 = 0
    for (x, y) in grid:
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x)
        y2 = max(y2, y)
    return [(x1, y1), (x2, y2)]

def drawRec(image, rec, height, width) :
    (x1, y1) = rec[0]
    (x2, y2) = rec[1]
    for x in range(height):
        for y in range(width) :
            if ((x == x1 or x == x2) and y >= y1 and y <= y2) or ((y == y2 or y == y1) and x >= x1 and x <= x2):
                image[x][y] = 255
    return image

def getOutLine(rec) :                  #获取轮廓的长度
    (x0, y0) = rec[0]
    (x1, y1) = rec[1]
    return (x1 - x0) * 2 + (y1 - y0) * 2



def drawSingle(height, width):                   #一个一个画出
    for i in range(len(anss)):                   # 对于每一个联通区域创建一个照片
        if i > 15 : break                        # 扔掉二维码
        grid = anss[i]
        image = draw(grid, height, width)
        print("该区域的面积:" + str(getAear(grid)))
        print("该区域的重心:" + str(getFoucs(grid)))
        print("该区域的最小平行轴外接矩形")
        rec = getRec(grid)
        print(rec)
        print("该区域的轮廓长度为:" + str(getOutLine(rec)))
        image = drawRec(image, rec, width, height)
        cv.imshow('Connection' + str(i), image)
        image = image[rec[0][0]:rec[1][0], rec[0][1]:rec[1][1]]
        cv.imshow("Connection_"+str(i), image)


def drawAll(height, width) :                         #画出所有的区域
    img = np.zeros((height, width), np.uint8)
    for grid in anss:
        rec = getRec(grid)
        image = draw(grid, height, width)
        image = drawRec(image, rec, height, width)
        for x in range(height):
            for y in range(width) :
                img[x][y] += image[x][y]
    cv.imshow('all', img)

def last() :
    image = cv.imread("image2.jpg", 0)  # 转化为单通道的灰度图
    thresh = mythresh(image)  # 阈值分割为二值图，使用大基算法得到的最优的阈值
    # save(thresh)
    # cv.imshow('thresh', thresh)
    findConnection(thresh)  # 找联通,并绘制相应的
    print('anss.len = ' + str(len(anss)))
    drawSingle(thresh.shape[0], thresh.shape[1])
    drawAll(thresh.shape[0], thresh.shape[1])
    # drawSingle2(thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    first()
    second()
    third()
    last()