
import numpy as np
def test(img) :
    x = 10
    y = 10
    for i in range(x):
        for j in range(y):
            if j % 2 == 0 :
                img[i][j] = 1
    return img



if __name__ == '__main__':
    img = np.zeros((10, 10), np.uint8)
    img = test(img)
    for line in img:
        for x in line:
            print(x, end=" ")
        print()