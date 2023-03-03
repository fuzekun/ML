from firstWEB.models import *
import cv2
from datetime import datetime
import time
def insert(request):
    # value_a = request.POST['valA']
    # value_b = request.POST['valB']
    # c = int(value_a) + int(value_b)
    # cal.objects.create(value_a=value_a, value_b=value_b, result=c)
    img_url = r"F\img\fire.jpg"
    vedio.objects.create(img_url=img_url)

def getAll():
    data_list = vedio.objects.all()
    return data_list



# 进行模糊查询，找到相应的图片
def getLikeImg(date) :
    time_str = '2021-04-24 13:24' #注意月份应该加上0
    date_form = '%Y-%m-%d %H:%M'
    datetime_obj = datetime.strptime(time_str, date_form)
    ret_stamp = int(time.mktime(datetime_obj.timetuple()))
    img_id = str(ret_stamp[0:8])

    # img_name='F:\\fire\\'+str(ret_stamp)+'.jpg'
    # data = vedio.objects.get(id=id)
    # img = cv2.imread(data.img_url)
    # img1 = cv2.imencode('.jpg', img)[1]