# from aip import AipFace
#
# """ 你的 APPID AK SK """
# APP_ID = '16223524'
# API_KEY = '7DFkKXBNV0iyUmUB8Vqqjnyi'
# SECRET_KEY = 'uZ6IpwaQsLnfEr6R001mWYAlzg9WyU5N'
#
# client = AipFace(APP_ID, API_KEY, SECRET_KEY)

 # encoding:utf-8
import requests
from aip import AipFace
import base64
import json

def createCli():
    """ 你的 APPID AK SK """
    APP_ID = '16223524'
    API_KEY = '7DFkKXBNV0iyUmUB8Vqqjnyi'
    SECRET_KEY = 'uZ6IpwaQsLnfEr6R001mWYAlzg9WyU5N'

    client = AipFace(APP_ID, API_KEY, SECRET_KEY)
    return client


def getAUTH():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=7DFkKXBNV0iyUmUB8Vqqjnyi&client_secret=uZ6IpwaQsLnfEr6R001mWYAlzg9WyU5N'
    response = requests.get(host)
    if response:
        print(response.json())
def faceRcn(face):
    client = createCli()
    # img = open('1.jpeg', 'rb').read()
    # bimg1 = base64.b64encode(img)
    # jimg1 = str(bimg1,'utf-8')
    # img = open('2.jpeg', 'rb').read()
    # bimg2 = base64.b64encode(img)
    # jimg2 = str(bimg2,'utf-8')

    imgpath = 'firstWEB\\utils\\1.jpg'
    try:
        result = client.match([
            {
                'image': face,
                'image_type': 'BASE64',
            },
            {
                'image': str(base64.b64encode(open(imgpath, 'rb').read()), 'utf-8'),
                'image_type': 'BASE64',
            }
        ])
        score = result['result']['score']
    except:
        print("出现异常")
        return 0
    else :
        return score

# if __name__ == '__main__':
#     print(faceRcn("fsa"))