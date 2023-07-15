from django.shortcuts import render
from django import views
from firstWEB.models import *
from firstWEB.models import danger as dang
from django.core import serializers
import json
from .models import vedio
from django.contrib.auth.models import User
# Create your views here.
from django.http import HttpResponse
from firstWEB.utils import authcode
from django.http import JsonResponse
from django.contrib import auth
import base64
from firstWEB.utils import faceRecgonize
from firstWEB.utils import faceDetect
import cv2
import os
import time
import numpy as np
from datetime import datetime
# from firstWEB.utils.fireDetect import smoke_file_obj
from firstWEB.service import dangerdb
from firstWEB.utils import fireDetect as fd
def index(request):
    return render(request, 'index.html')


def calPage(request):
    return render(request, 'cal.html')


def calculate(request):
    if request.POST:
        dangerdb.insert(request)
        return render(request, 'result.html', context={'result': 1})
    else :
        return HttpResponse('请使用post请求来申请本网页')

def calList(request):
    img_list = vedio.objects.filter(img_id__startswith="161934")
    # for img in img_list:
    #     print(img.img_id)
    # data_list = cal.objects.all()
    # for data in data_list:
    #     print(data.value_a, data.value_b, data.result)
    ret1_list = serializers.serialize("json", img_list)
    ret2_list = json.dumps(ret1_list)
    return render(request, 'rslt.html', {'img_list': ret2_list})


def delList(request):
    if request.POST:
        vedio.objects.all().delete()
        danger.objects.all().delete()
        return HttpResponse('删除成功')
    else :
        return  HttpResponse('请使用post请求访问本网页')

class Register(views.View):
    def get(self, request):
        return render(request, "register.html", context={"status": 0})
    def post(self, request):
        email = request.POST.get("email")
        password = request.POST.get("password")
        spassword = request.POST.get("spassword")
        terms = request.POST.get("terms")
        if terms == None:
            return render(request, "register.html", context={"status" : 2})

        if password == spassword:
            # 创建用户
            user = User.objects.create_user(email, password, email)
            user.save()
            return render(request, "login.html")
        else :
            return render(request, "register.html", context={"status" : 1})

# 用户登录视图类
class Login(views.View):
    def get(self, request):
        # get请求返回登录页面
        return render(request, "login.html")

    def post(self, request):
        data = request.POST
        # 获取用户登录信息
        authcode = data.get("authcode")
        username = data.get("username")
        password = data.get("password")
        print("输入验证码： ", authcode)
        # 验证码不正确
        if request.session.get("authcode").upper() != authcode.upper():
            print("验证码错误")
            return JsonResponse({"status": "1"})
        else:
            # 使用django的auth模块进行用户名密码验证
            print(username, password)
            user = auth.authenticate(username=username, password=password)
            if user:
                # 将用户名存入session中
                print("用户登录成功")
                request.session["user"] = username

                auth.login(request, user)  # 将用户信息添加到session中
                return JsonResponse({"status": "2"})
            else:
                print("用户认证失败")
                return JsonResponse({"status": "3"})


# 验证码视图类
class GetAuthImg(views.View):
    """获取验证码视图类"""

    def get(self, request):
        data = authcode.get_authcode_img(request)
        #print("验证码：", request.session.get("authcode"))
        return HttpResponse(data)

# 通向人脸识别认证登录页面
def faceR(request):
    return render(request, 'faceRcn.html')

def getImg(requet):
    img = requet.FILES['file']
    #print("读取完成")
    content = img.read()
    #print("二进制转换完成")
    bct = base64.b64encode(content)
    #print("bse64编码完成")
    face =str(bct, 'utf-8')
    #print("转换成字符串完成")
    score = faceRecgonize.faceRcn(face)
    print("成绩为:", score)
    result = {'score': score}
    return JsonResponse({'result': result})


    # img = requet.POST.get('file')
    # print(img)
    # file = requet.POST['file']
    # print(file)
    # postBody = requet.body
    # print(type(postBody))

    # files = requet.FILES.getlist('file')
    # for f in files:
    #     #保存了文件
    #     dest = open('/temp/' + f.name, 'wb+')
    #     str(base64.b64encode(open('1.jpg', 'rb').read()), 'utf-8')
    #     for chunk in f.chunks():
    #         dest.write(chunk)
    #     dest.close()
    #     print(f.name)
def getImgH(requst):
    return  render(requst, 'getImg.html')

def recgnize(request):
    return render(request, 'recgnize.html')

def getSolveImg(request):
    """
    :param request:
    :return:
    将图片进行人脸识别后保存到./firstWEB/static/img文件夹下
    """
    #print("进入图片处理")
    #读取图片
    img = request.FILES['file']
    #content = img.read();
    #处理图片，并且使用流进行图片的返回
    content = faceDetect.solveImg(img)
    import time
    # 保存图片
    t = time.time()
    name = int(round(t * 1000))
    img_name='./firstWEB/static/img/'+str(name)+'.jpg' #由于可以直接使用static目录下的内容，所以直接写成static就行
    # print(img_name)
    cv2.imwrite(img_name, content)
    ve = vedio()
    # 用时间作为id
    ve.img_id = str(name)
    # 保存url
    saveURL = "/static/img/"+str(name)+'.jpg'
    ve.img_url = saveURL
    ve.save()

    # 经过编码后得到的图片
    img1 = cv2.imencode('.jpg', content)[1]
    back_2 = base64.b64encode(img1)
    return HttpResponse(back_2)

"""
需要返回一个图片，但是传进来的是一个读取后的文件，传输出去的是一个Numpy的数组。
"""
def surveillance(request) :
    return render(request, 'surveillance.html')


def fireDetect(request):
    return render(request, "fireDetect.html")

def getFireImg(request):

    """"
    函数用来处理火焰，
    也就是处理危险。
    1. 如果有火焰，就写入数据库，所以，方法的返回值应该不仅仅是一个图片，应该返回一个原来的数组。
    2. 返回一个原来的数组之后，进行相应的图片处理，之后本系统决定是否保存图片，原图放在本系统的项目之中。
    3. 使用相对路径，不要使用绝对路径。
    4. 测试一下速度慢是因为调用还是因为运行。->已经解决，没有系统调用了

    """
    #获取并处理成可以保存的图片
    img = request.FILES['file']
    img_b = base64.b64encode(img.read())
    imD = base64.b64decode(img_b)
    nparr = np.fromstring(imD, np.uint8)
    sv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #处理图片，并且使用流进行图片的返回
    #以时间为图片命名
    t = time.time()
    name = int(round(t * 1000))
    content = fd.getSolveImg(sv, name)

    #经过编码后得到的图片
    img1 = cv2.imencode('.jpg', content)[1]
    back_2 = base64.b64encode(img1)
    return HttpResponse(back_2)





def getVedio(request):
    return render(request, "getRVedio.html")

#获取回放
def getRvedio(request):
    # 获取事件字符串
    dt = request.POST.get('date')
    tm = request.POST.get("time")
    if len(str(tm)) != 0:
        time_str = dt + " " + tm + ":00"
    else :
        time_str = dt + " " + "00:00:00"
    # time_str = "2021-04-27 17:56:15"
    # print(time_str) #看月份有没有加上0
    # 将time_str转换为时间戳，以开始时间进行模糊搜索，取钱8位,显示一分钟左右的视频
    date_form = '%Y-%m-%d %H:%M:%S'
    datetime_obj = datetime.strptime(time_str, date_form)
    ret_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    #测试使用前6位,默认使用8位
    # t = time.time()
    # rt = str(t)[0:6]
    if len(str(tm)) != 0 :
        ret_stamp = str(ret_stamp)[0:8]
    else :
        # print("使用5位")
        ret_stamp = str(ret_stamp)[0:5]
    img_list = vedio.objects.filter(img_id__startswith=str(ret_stamp))
    # for img in img_list:
    #     print(img.img_id)
    # data_list = cal.objects.all()
    # for data in data_list:
    #     print(data.value_a, data.value_b, data.result)
    ret1_list = serializers.serialize("json", img_list)
    ret2_list = json.dumps(ret1_list)
    return HttpResponse(ret2_list)
    #将传入的时间转化为ms级别的时间,字符串应该默认加上
    # tail = '.001'
    # time_str = '2021-04-24 13:24:25' #注意月份应该加上0
    # date_form = '%Y-%m-%d %H:%M:%S.%f'
    # datetime_obj = datetime.strptime(time_str, date_form)
    # ret_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    # img_name='F:\\fire\\'+str(ret_stamp)+'.jpg'
    # 以下是使用id进行查询
    # data = vedio.objects.get(id=id)
    # img = cv2.imread(data.img_url)
    # img1 = cv2.imencode('.jpg', img)[1]
    # back = base64.b64encode(img1)
    # return HttpResponse(back)

    """
    方法1
        传递两个参数：
        1.开始时间->找到idb (使用模糊查询)
        2.结束时间->找到ide
        3.[idb, ide]之间的所有图片进行一次性的返回
        4.前端进行图像的轮播,使用js对数组中的图像进行显示
            4.1 写一个显示图像的函数，其中的数组下标设为全局变量
            4.2 每隔200ms调用函数
    方法2
        1.使用html拍摄一段视频上传
        2.保存在本地之后，将url保存在数据库中
        3.传递一个url进行回放
    方法1改进:
        1.传递一个参数
        2.使用模糊查询找到时间(精确到分钟，找最近一小时的时间)
        3.
    """
    # for data in data_list:
    #     img = cv2.imread(data.img_url)
    #     if(data.id == 58):
    #         img1 = cv2.imencode('.jpg', img)[1]
    #         back_2 = base64.b64encode(img1)
    #         return HttpResponse(back_2)

    # 方法二
    # vedio_name = 'test.av'
    # return HttpResponse("media/vedio" + vedio_name)#放在media中,返回url即可

def calenda(request) :
    return  render(request, 'canlenda.html')


def cameraSv(request):
    ca = camera()
    ca.id = 3
    camera.Ip = "192.168.0.1"
    ca.save()
    return HttpResponse("保存完成")

def fireP(request):
    danger_list = danger.objects.filter(img_name__startswith="/static") #获取所有在static下的危险信息
    return render(request, 'firep.html', {'danger_list': danger_list})

def cgdanger(request):
    id = request.POST.get('id')
    # print(id)
    # print(type(id))
    dang.objects.filter(danger_id=int(id)).update(progress='2')
    obj = dang.objects.get(id=int(id))
    obj.progress = '2'
    print(obj)
    obj.save()
    danger_list = danger.objects.filter(img_name__startswith="/static") #获取所有在static下的危险信息
    return render(request, 'firep.html', {'danger_list': danger_list})
