# def mythresh(image, thresh = 135) :
#     img = []
#     for line in image:
#         tmp = [0 if s > thresh else 255 for s in line]
#         img.append(tmp)
#     img = np.array(img, np.uint8)
#     return img
#
# def save(value) :            # 将像素数组放入文件中
#     print("文件保存中...")
#     file = open("pixel3.txt", 'w')
#     x = 0
#     for line in value:
#         y = 0
#         for v in line :
#             file.write('(' + str(x) + ',' + str(y) + '):' + str(v) + ' ')
#             y += 1
#         x += 1
#         file.write('\n')
#     print("保存完成")
# def horizontalCut(img):
#     (x, y) = img.shape
#     cnt = np.zeros(x)
#     for i in range(x):
#         for j in range(y):
#             if img[i, j] == 255:              #白色像素255，代表内容
#                 cnt[i] += 1
#     plt.plot(range(x), cnt)
#     plt.show()
#     start=[]                                 #开始索引数组
#     end=[]                                   #结束索引数组
#     # print("y / 20", y / 20)
#     if cnt[0] != 0 : start.append(0)         #开始边界
#     for index in range(1, x - 1):
#         #本行大于0，上一行等于0，即开始
#         # print(cnt[index])
#         if cnt[index] != 0 and cnt[index - 1] == 0:
#             start.append(index)
#         #本行大于0，下一行等0，即结束
#         elif cnt[index] != 0 and cnt[index + 1] == 0:
#              end.append(index)
#     if cnt[x - 1] != 0 : end.append(x - 1)      #结束边界
#     imgs = []
#     for i in range(min(len(start), len(end))):
#         print("start = " + str(start[i]))       #输出为了防止start > end
#         print("end = " + str(end[i]))
#         imgi = img[start[i]:end[i]]
#         imgs.append(imgi)
#         show("r_" + str(i), imgi)
#     return imgs

# def getRec(grid) :   #得到最小的平行外界矩形
#     x1 = sys.maxsize        #左上角和右下角
#     y1 = sys.maxsize
#     x2 = 0
#     y2 = 0
#     for (x, y) in grid:
#         x1 = min(x1, x)
#         y1 = min(y1, y)
#         x2 = max(x2, x)
#         y2 = max(y2, y)
#     return [(x1, y1), (x2, y2)]
#
#
# def getSingle(image, rec) :  #根据占比多少判断是否为矩形
#     (x1, y1) = rec[0]
#     (x2, y2) = rec[1]
#     img = image[x1:x2, y1:y2]
#     return img





# def drawSingle(image):                                   #一个一个画出
#     n = len(anss)
#     tpo = 0
#     bottom = 0
#     for i in range(1, n):                               # 第一个联通的是边界，从第二个开始
#         rec = getRec(anss[i])
#         img = getSingle(image, rec)
#         # show("component_" + str[i], img)
#         if i == 1:
#             top = rec[0][0]
#             bottom = rec[1][0]
#         else :
#             tbottom = rec[1][0]
#             if tbottom < bottom * 1.5:                   #去字母区域的最大下界
#                 # print("tbottom = " + str(tbottom))
#                 bottom = max(bottom, tbottom)
#     print("top = " + str(top) + " bottom = " + str(bottom))
#     return (top, bottom)


#
# def columnCut(img):                         #图片进行列分割
#     (x, y) = img.shape
#     cnt = np.zeros(y)
#     for j in range(y):
#         for i in range(x):
#             if img[i, j] == 255:              #白色像素255，代表内容
#                 cnt[j] += 1
#     # plt.plot(range(x), cnt)
#     # plt.show()
#     start=[]                                 #开始索引数组
#     end=[]                                   #结束索引数组
#     if cnt[0] != 0 : start.append(0)         #开始边界
#     for index in range(1, y - 1):
#         #本列大于0，上一列等于0，即开始
#         if cnt[index] != 0 and cnt[index - 1] == 0:
#             start.append(index)
#         #本列大于0，下一列等0，即结束
#         elif cnt[index + 1] == 0 and cnt[index] != 0:
#              end.append(index)
#     if cnt[y - 1] != 0 : end.append(x - 1)      #结束边界
#     imgs = []
#     print("col.len = "  + str(min(len(start), len(end))))
#     for i in range(min(len(start), len(end))):
#         print("c_start = " + str(start[i]))       #输出防止start > end
#         print("c_end = " + str(end[i]))
#         if start[i] >= end[i] :
#             continue
#         imgi = img[:, start[i]:end[i]]
#         imgs.append(imgi)
#         imgi = solveWImg(imgi)
#         if len(imgi) > 100:                       #有可能截取出来噪声点
#             show("c_" + str(i), imgi)
#         # cv.imwrite("numbers\\"
#         #            + str(i) + ".jpg", imgi)
#     return imgs


# def cutLine(img) :          #使用bfs进行列切割
#     findConnection(img, 50)

# def draw(image) :
#     show('changeTo0', image)
#     recs.sort(key=lambda x:(x[0], x[2]))         #根据列进行排序,之后根据列
#     print(len(recs))
#     for i in range(len(recs)):
#         rec = recs[i]
#         # print("Rec")
#         # for i in rec:
#         #     print(i, end = " ")
#         # print()
#         img = image[rec[0]: rec[1], rec[2]: rec[3]]
#         show('char_' + str(i), img)