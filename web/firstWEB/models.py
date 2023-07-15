from django.db import models




# Create your models here.
# 创建数据库的表
class cal(models.Model):
    value_a = models.CharField(max_length=10)
    value_b = models.CharField(max_length=10)
    result = models.CharField(max_length=10)

class areas(models.Model) :
    areas_id = models.IntegerField
    areas_name = models.CharField(max_length=10)
    remarks = models.CharField(max_length=10)
    director = models.CharField(max_length=10)#默认区域负责人就一个


class camera(models.Model) :
    Ip = models.CharField(max_length=20)
    camera_type = models.IntegerField
    area_id = models.IntegerField
#视频，使用图片进行存储,时间作为id
class vedio(models.Model):
    img_id = models.CharField(max_length=20)
    img_url = models.CharField(max_length=20)

class danger(models.Model):
    danger_id = models.CharField(max_length=30)
    danger_type = models.CharField(max_length=30)
    solver = models.CharField(max_length=30)# 直接使用处理人的姓名，不用id了否则还得做连接
    area_name = models.CharField(max_length=30) #同上
    camera_id = models.IntegerField
    progress=models.CharField(max_length=30)
    img_name = models.CharField(max_length=30)
