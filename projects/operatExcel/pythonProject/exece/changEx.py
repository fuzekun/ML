from xlutils.copy import copy
import xlrd
import os
import sys

print("要替换的文件需与脚本同一目录")
a = input("输入替换前的内容 \n")
b = input("输入替换后的内容 \n")
c = input("输入替换是第几列 \n")
c = int(c) - 1

#当前路径 print(sys.path[0])
for files in os.walk(sys.path[0]):
    for file in files[2]:
        #print(file[-3:])
        if(file[-3:] == 'xls'):
            #1、打一要修改的excel
            #2、再打开另一个excel
            #3、把第一个excel里面修改东西写到第二个里头
            #4、把原来的excel删掉，新的excel名改成原来的名字
            book = xlrd.open_workbook(file)
            table1 = book.sheets()[0]
            nr = table1.nrows
            nc = table1.ncols
            num = 0
            #复制一个excel
            new_book = copy(book)
            #通过获取到新的excel里面的sheet页
            sheet = new_book.get_sheet(0)
        for i in range(0,nr):
#print(str(table1.cell(i,c).value))
            if(str(table1.cell(i,c).value) == a):
                sheet.write(i, c, b)
num = num + 1
#保存新的excel，保存excel必须使用后缀名是.xls的，不是能是.xlsx的
new_book.save('2.xls')
#删除原有
os.remove(file)
#将新文件重命名原有文件
os.rename('2.xls',file)
print(file + " 替换完成 共替换"+ str(num) + "条")

ex = input("是否退出,Y or N ? \n")
if(ex == 'Y'):
    exit()