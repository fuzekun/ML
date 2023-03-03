from django.urls import path

from . import views

urlpatterns = [
    path('index', views.index, name = 'index'),
    path('cal', views.calPage, name = 'cal'),
    path('calculate', views.calculate, name = 'calculate'),
    path('calList', views.calList, name = 'calList'),
    path('delList', views.delList, name = 'delList'),
    path('register', views.Register.as_view(), name = 'register'),
    path('get_auth_img', views.GetAuthImg.as_view(), name = 'get_auth_img'),
    path('login', views.Login.as_view(), name = 'login'),
    path('faceR', views.faceR, name = 'faceR'),
    path('getImg', views.getImg, name = 'getImg'),
    path('getImgh',views.getImgH, name = 'getImgh'),
    path('recgnize', views.recgnize, name = 'recgnize'),
    path('getSolveImg', views.getSolveImg, name = 'getSolveImg'),
    path('surveillance', views.surveillance, name = 'surveillance'),
    path('getFireImg', views.getFireImg, name = 'getFireImg'),
    path('fireDetect', views.fireDetect, name='fireDetect'),
    path('getVedio', views.getVedio, name='getVedio'),
    path('getRvedio', views.getRvedio, name='getRvedio'),
    path('calenda', views.calenda, name = 'calenda'),
    path('cameraSv', views.cameraSv, name = 'cameraSv'),
    path('fireP', views.fireP, name = 'fireP'),
    path('cgdanger', views.cgdanger, name = 'cgdanger'),
]