import cv2

from firstWEB.models import *
import os
#进行图片的保存
"""
    1.放入f:/fire/
    2.将url存入danger数据库
    
    img就是打好了label的图片，url就保存的路径,默认为f:/fire/img_name(时间可以写成当前的时间，不用传来的时间)
"""
def save_fire_img(img, name) :
    da = danger()
    da.solver = "fuzekun"
    da.progress = '1'
    da.area_name = "hall"
    da.danger_type = '1'   # 这个是使用火焰检测的id
    da.camera_id = 1
    save_url = "/static/fire/"+str(name)+'.jpg'   #数据库中的url
    da.img_name = save_url
    da.danger_id = name
    da.save()
    img_url = "./firstWEB/static/fire/"+str(name)+'.jpg'#原文件中的url
    cv2.imwrite(img_url, img)



# 给图片打上标签
def label(img, ls, name): # 图，数组,路径
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
    if (flag):  # 有火焰，保存图片
        save_fire_img(img, name)

    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    return img

# 进行调用进程
def getSolveImg(img, name) : # 图片就是前端的图片， img_name是刚获取图片之后的保存路径(文件读取路径)，新的保存路径应该放在fire里面
    # img = cv2.imread(img_name) #要么直接读取，要么经过编码,这里是直接读取
    # 想一想怎么改成相对路径
    img_name = 'F:\\fireD\\' + str(name) + '.jpg' #读取路径
    # path = os.getcwd()
    # print("path:", path)
    # cmd = "python " + path + "\\Smoke_Fire_Detection-main\\smoke_file_obj.py --img_name " + img_name
    # print("cmd", cmd)
    cmd = "python Smoke_Fire_Detection-main\\smoke_file_obj.py --img_name " + img_name
    # print(cmd)
    a = os.popen(cmd)
    ret_back = a.read()
    # 过滤掉前面的Flusing layers...
    content = ret_back[18:]
    print(content)
    # 转化为字典列表
    arr = eval(content)
    # print(arr)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    ret_img = label(img, arr, name)
    # print(type(ret_img))
    return ret_img
    # solveFireImg(img, arr)