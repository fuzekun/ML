import cv2
import base64
import numpy as np
import time

def solveImg(img):
    #print(type(img))
    img_b = base64.b64encode(img.read())
    imD = base64.b64decode(img_b)
    nparr = np.fromstring(imD, np.uint8)
    # cv2.IMREAD_COLOR 以彩色模式读入 1
    # cv2.IMREAD_GRAYSCALE 以灰色模式读入 0
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    color = (0, 255, 0)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print("当前路径为:", os.getcwd())
    classfierp = "firstWEB\\utils\\haarcascade_frontalface_default.xml"
    # classfierp = "haarcascade_frontalface_default.xml"
    classfier = cv2.CascadeClassifier(classfierp)
    faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects) > 0:
        for faceRects in faceRects:
            x, y, w, h = faceRects
            cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)
    # cv2.imshow("Find Faces", img)
    # cv2.waitKey(0)
    #print(type(img))
    return img
if __name__ == '__main__':
    img = open('1.jpg', 'rb')
    content = solveImg(img)
    #print(type(content))
    t = time.time()
    name = int(round(t * 1000))
    img_name='./'+str(name)+'.jpg'

    # cv2.imwrite(img_name, content)
