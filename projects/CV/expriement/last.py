import cv2 as cv
import numpy as np
import sys

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



def drawSingle(height, width):                                   #一个一个画出
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
        image = drawRec(image, rec, thresh.shape[0], thresh.shape[1])
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




if __name__ == '__main__':
    image = cv.imread("image2.jpg", 0)  # 转化为单通道的灰度图
    thresh = mythresh(image)           # 阈值分割为二值图，使用大基算法得到的最优的阈值
    # save(thresh)
    # cv.imshow('thresh', thresh)
    findConnection(thresh)             # 找联通,并绘制相应的
    print('anss.len = ' + str(len(anss)))
    drawSingle(thresh.shape[0], thresh.shape[1])
    drawAll(thresh.shape[0], thresh.shape[1])
    # drawSingle2(thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()