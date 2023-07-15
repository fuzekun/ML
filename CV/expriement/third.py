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
    cv.imshow("Color Edge", dst)

def test1() :               #库 的高斯滤波 与 边缘检测
    src = cv.imread('images/ISBN 978-7-300-23418-2.jpg')
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
if __name__ == '__main__':
    # blur()
    # test2()
    test1()
