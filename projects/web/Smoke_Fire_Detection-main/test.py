import cv2
import  os
import  time
# 返回框好的图片
def solveFireImg(img, ls): # 图片，路径，数组
    # print(ls)
    flag = False
    for dic in ls:
        flag = True
        if (dic['label'] == 'fire'):
            rec = dic['bbox']
            cv2.rectangle(img, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 255, 0), 3)
        if (dic['label'] == 'smoke'):
            rec = dic['bbox']
            cv2.rectangle(img, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 0, 255), 3)
    # if (flag):  # 有火焰，保存图片
    #     cv2.imwrite(img_name, img)  # 使用传入的参数自动覆盖原来的图片

    cv2.imshow("test", img)
    cv2.waitKey(0)
    return img


# 返回一长处理好的图片（保存在本地）
def test1():
    # 读取输出的内容
    t = time.time()
    name = int(round(t * 1000))
    img_name = 'F:/fire/fire.jpg'
    cmd = "python F:\\各种文件\\pythonFile\\Web\\Smoke_Fire_Detection-main\\smoke_file_obj.py --img_name " + img_name
    print(cmd)
    a = os.popen(cmd)
    img = a.read()
    # 过滤掉前面的Flusing layers
    content = img[18:]
    print(content)
    # 转化为base64编码之后的图片
    # print(content)
    byte1 = bytes(content, encoding='utf-8')
    print("转化为64编码后的图片为:")
    print(byte1)
# 返回需要标记的数组
def test2(img_name) :

    cmd = "python F:\\各种文件\\pythonFile\\Web\\Smoke_Fire_Detection-main\\smoke_file_obj.py --img_name " + img_name
    print(cmd)
    a = os.popen(cmd)
    img = a.read()
    # 过滤掉前面的Flusing layers
    content = img[18:]
    print(content)
    # 转化为字典列表
    arr = eval(content)
    print(arr)
    img = cv2.imread(img_name)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    solveFireImg(img, arr)
    # solveFireImg(img, arr)
if __name__ == '__main__':
    img_name = 'F:/fire/noFire.jpg'
    test2(img_name)


"""
    如果使用该系统，
    1.需要在系统的f盘下建立，fire和img两个目录
    2.需要数据库的支持
    3.需要model模块的支持，需要在指定目录下将model模块放入
    
    否则无法启动火焰检测和回放功能。
    当然如果没有数据库会报错。
    如果没有文件夹还可以自己进行建立，或者给本系统权限，由本系统进行建立。
"""

