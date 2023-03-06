"""
@author:fuzekun
@file:check_sleepy.py
@time:2023/03/03
@description:
1. 剪切
2. 统计

"""
import os.path
from moviepy.video.io.VideoFileClip import VideoFileClip
import pandas as pd
from camera.check_sleepy import check_sleepy



path = 'd:/data/camera/'                                                            # 待切割视频存储目录
file_name = 'WIN_20221118_09_53_21_Pro.mp4'


"""
如果获取全部, 就获取每一次的眨眼，点头，哈欠，perclose等信息
但如果想快点，还是直接检测到困，就直接停下来，不困继续检测。
"""
# get_other_message = 0                                                               # 是否获取每一分钟的全部信息


"""
是否展示，也就是出现那个视频，展示的时候，展示，运行的时候就不用了
"""
show_video = 0                                                                      # 展示的时候，可以使用test_run(), 然后show_video = 1;




def run():

    save_path = 'd:/data/camera/'
    save_name = file_name + '_label.csv'                                                # 文件名称就是 受试者_label.csv
    source_file = path + file_name
    source_video = VideoFileClip(source_file)
    total_sec = int(source_video.duration)
    list = []  # 存储每一分钟的结果
    field_name = ['flag', 'perclose', 'eye_total', 'mouse_total', 'node_total']
    try:
        for i in range(0, min(total_sec, 3600), 60):                                         # 截取一小时
            start_time = i
            stop_time = min(total_sec, i + 60)
            if stop_time - start_time < 60:                                                     # 不到1min,不进行截取
                break
            video = source_video.subclip(int(start_time), int(stop_time))  # 执行剪切操作
            target_file = path + "tmp" + ".mp4"                                                 # 临时文件的地址和初始文件的地址相同，名字叫做tmp
            video.write_videofile(target_file)                                                  # 保存视频
            print(f"---------------正在处理第{i // 60}到第{i // 60 + 1}分钟钟的视频---------------")
            ans = check_sleepy(target_file, show_video)
            list.append(ans)
            print(f"-------------第{i // 60}到第{i // 60 + 1}分钟的视频处理完成------------------")
        # 如果最后完成，重新写入一次，这个是因为，追加方式的序号都是0
        ans = pd.DataFrame(columns=field_name, data=list)
        ans.to_csv(os.path.join(save_path, save_name))
    except Exception as e :
        print(e)
        print("程序异常终止")
    finally:
        print('--------结果保存------')
        # 如果最后完成，重新写入一次，这个是因为，追加方式的序号都是0
        print("------------  最终结果为 ------------")
        print(list)
        ans = pd.DataFrame(columns=field_name, data=list)
        ans.to_csv(os.path.join(save_path, save_name))


def test_show():
    """
    展示的时候，跑这个代码
    """
    global show_video
    show_video = 1
    run()

def test_total_len():
    """
    测试截取视频，最后一段会不会报错
    """
    path = 'd:/data/camera/'  # 待切割视频存储目录
    file_name = 'WIN_20221118_09_53_21_Pro.mp4'
    source_file = path + file_name
    source_video = VideoFileClip(source_file)
    total_sec = int(source_video.duration)
    for i in range(0, min(total_sec, 3600), 60):                        # 超过1小时，截取一小时
        start_time = i
        stop_time = min(total_sec, i + 60)
        if stop_time - start_time < 60:                                 # 最后不到1min，的不进行截取，也就是不到1h，最后一段舍弃。
            break
        video = source_video.subclip(int(start_time), int(stop_time))  # 执行剪切操作
        print(f"---------------正在处理第{i}到第{i + 60}分钟的视频---------------")
        print(f"-------------第{i}分钟到第{i + 60}的视频处理完成------------------")



def test_read_write() :
    list = [[1,2,3,3,4]]
    field_name = ['flag', 'perclose', 'eye_total', 'mouse_total', 'node_total']
    save_path = 'd:/data/camera/'
    save_name = 'label.csv'
    ans = pd.DataFrame(columns=field_name, data=list)
    ans.to_csv(os.path.join(save_path, save_name), mode='a', header=False)

def test_check() :
    target_file = 'd:/data/camera/tmp.mp4'  # 待切割视频存储目录
    ans = check_sleepy(target_file, show_video)

if __name__ == '__main__':
    run()
