# 项目分解



## 功能



### 用户以及权限管理



1. 人脸对比，使用了百度api进行人脸的识别。

扩展性不好，使用包进行，使用监听者模式，注册一个类，然后接受消息队列的通知。

2. 用户登录注册，以及访问拦截器等。



### AI模型管理

主要为了扩展性，随着AI的发展，需要能够动态添加模块。



1. 火焰检测

将系统调用修改掉-路径是相对于当前项目文件夹的。

2. 模型缓存

将人脸和火焰检测的模型进行一个缓存

-  下载django-redis

```java
pip install django-redis
```

- 安装完成后，将其添加到INSTALLED_APPS中(settings.py)：

```python
INSTALLED_APPS = [
    # ...
    'django_redis',
    # ...
]
```

- 在settings.py文件中，配置Redis作为缓存后端。

```python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://localhost:6379/0',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
    },
}
```

- 模型缓存

```java
from django.core.cache import cache
from myapp.models import MyModel

my_objects = MyModel.objects.all()
cache.set('my_model', list(my_objects), timeout=3600)  # 缓存60分钟
```

- 模型获取

```java
from django.core.cache import cache
from myapp.models import MyModel

my_objects = cache.get('my_model')
if my_objects is None:
    my_objects = MyModel.objects.all()
    cache.set('my_model', list(my_objects), timeout=3600)  # 缓存60分钟
```





### 视频管理

1. 视频回放

视频回放功能出现了bug，因为前6位并不表示年月，所以就是是那一天也不能找到。

2. 视频存储





### 设备管理

1. 摄像头管理
2. 内存管理



### 检测结果以及处理管理

对于检测到的结果需要进行一个系统通知，如果处理完成，需要进行销毁操作。