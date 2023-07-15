from datetime import  datetime
import time
def getLikeImg(date) :
    """
    差100就是差1分40秒
    所以一次可以获得1分40秒的视频
    startwith(前8位)
    如果是7位可以截取
    1000 / 60 = 100 / 6 = 16.7分钟的视频
    """
    time_str = '2021-04-24 13:23:00' #注意月份应该加上0
    date_form = '%Y-%m-%d %H:%M:%S'

    datetime_obj = datetime.strptime(time_str, date_form)
    ret_stamp = int(time.mktime(datetime_obj.timetuple()))
    print(str(ret_stamp)[0:8])
    # img_name='F:\\fire\\'+str(ret_stamp)+'.jpg'
    # data = vedio.objects.get(id=id)
    # img = cv2.imread(data.img_url)
    # img1 = cv2.imencode('.jpg', img)[1]
if __name__ == '__main__':
    getLikeImg("test")