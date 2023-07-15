from django.db import models

# Create your models here.
from django.db import models
class danger(models.Model):
    danger_id = models.IntegerField
    danger_type = models.IntegerField
    solver = models.CharField(max_length=10)# 直接使用处理人的姓名，不用id了否则还得做连接
    area_name = models.CharField(max_length=10) #同上
    camera_id = models.IntegerField
    progress=models.IntegerField
    def __str__(self):
        return self.danger_type
    class Meta:
        # 设置模型所属的APP，在数据库DB1中生成数据表
        # 若不设置app_label,则默认在当前所在的APP
        app_label = "fireDetect"
        db_table='danger'           #自定义数据表名称
        verbose_name = '危险情况表'  #城市信息表

class user(models.Model):
    username = models.CharField
    password = models.CharField