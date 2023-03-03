import argparse

import  base64
from models.experimental import *
from utils.datasets import *
from utils.general import *
from utils import torch_utils
import sys


class Smoke_File_Detector():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=r'Smoke_Fire_Detection-main\weights\smoke.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--img_name', type=str)
        self.opt = parser.parse_args()

        self.device = torch_utils.select_device(self.opt.device)

        # Initialize
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load models
        self.model = attempt_load(self.opt.weights, map_location=self.device)
        self.imgsz = check_img_size(self.opt.img_size, s=self.model.stride.max())
        if self.half:
            self.model.half()

    # 本地调用
    def detect_test(self,test_list):
        for i,img in enumerate(test_list):
            im0 = img
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)  # faster

            # Run inference
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            if i == 0:
                batch_img = img
            else:
                batch_img = torch.cat([batch_img,img],axis = 0)

        pred = self.model(batch_img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        # Process detections
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        batch_results = []
        for i, det in enumerate(pred):  # detections per image
            results = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(batch_img.shape[2:], det[:, :4], im0.shape).round()
                device = ('cuda' if torch.cuda.is_available() else 'cpu')
                det = det.to(device).data.cpu().numpy()
                for *xyxy, conf, cls in det:
                    w = xyxy[2]-xyxy[0]
                    h = xyxy[3]-xyxy[1]
                    result = {'bbox':xyxy, 'label':names[int(cls)], 'conf':conf}
                    results.append(result)

            batch_results.append(results)

        # print(batch_results)
        return batch_results

    # server调用
    def detect(self, **kwargs):
        params = kwargs
        test_list = [params["img"]]
        for i,img in enumerate(test_list):
            im0 = img
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # Run inference
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            if i == 0:
                batch_img = img
            else:
                batch_img = torch.cat([batch_img,img],axis = 0)

        pred = self.model(batch_img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        # Process detections
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        batch_results = []
        for i, det in enumerate(pred):  # detections per image
            results = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(batch_img.shape[2:], det[:, :4], im0.shape).round()
                device = ('cuda' if torch.cuda.is_available() else 'cpu')
                det = det.to(device).data.cpu().numpy()
                for *xyxy, conf, cls in det:
                    w = xyxy[2]-xyxy[0]
                    h = xyxy[3]-xyxy[1]
                    result = {'bbox':xyxy, 'label':names[int(cls)], 'conf':conf}
                    results.append(result)

            batch_results.append(results)

        # print(batch_results)
        return batch_results

    # 返回框好的图片
    def solveFireImg(self, img, img_name):
        ls = self.detect_test([img])[0]
        # print(ls)
        flag = False
        for dic in ls:
            flag = True
            if(dic['label'] == 'fire'):
                rec = dic['bbox']
                cv2.rectangle(img, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 255, 0), 3)
            if(dic['label'] == 'smoke'):
                rec = dic['bbox']
                cv2.rectangle(img, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0, 0, 255), 3)
        if(flag) : #有火焰，保存图片
            cv2.imwrite(img_name, img) # 使用传入的参数自动覆盖原来的图片
        return img




if __name__ == "__main__":
    import cv2
    img_name = sys.argv[2]
    img = cv2.imread(img_name)

    det = Smoke_File_Detector()
    arr = det.detect_test([img])[0]
    print(arr)
    # content = det.solveFireImg(img, img_name)
    # # cv2.imshow('test', content)
    # # cv2.waitKey(0)
    # # 保存图片
    # t = time.time()
    # # name = int(round(t * 1000))
    # # img_name='F:\\img\\'+str(name)+'.jpg'
    # img1 = cv2.imencode('.jpg', content)[1]
    # back_2 = base64.b64encode(img1)
    # back_3 = str(back_2, encoding="utf-8")
    # print(back_3, end="")



    """以下的内容一定要注释掉"""
    #测试保存的明亮航参数保存图片
    # img_name = sys.argv[2]
    # print(type(back_2))
    # print("图片")
    # print(img_name)
    # cv2.imwrite(img_name, content)

    # 测试如果直接输出会怎么样





