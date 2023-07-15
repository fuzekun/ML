import xlrd
from xlutils.copy import copy
"""
    1. 保证排序之后
    2. 保证没有浮点数作为名字
    3. 修改后会改变文件的字体的格式
"""
def readEx(fileName, sheetId, c) :
    names = []
    worksheet = xlrd.open_workbook(fileName)
    sheet_names = worksheet.sheet_names()
    # 返回sheet的name ['structure', 'data', 'Sheet1', '备注', '命运卡', '地图（废弃）']
    print(sheet_names)
    # 取第二个sheet页data
    rsheet = worksheet.sheet_by_index(sheetId)
    # row表示行
    for row in rsheet.get_rows():
        # 获取第c列
        id_column = row[c]
        id_value = id_column.value
        names.append(str(id_value).split('.')[0])
    return names

def write(fileName, sheetId, names, c) :

    worksheet = xlrd.open_workbook(fileName)
    new_book = copy(worksheet)
    sheet_names = worksheet.sheet_names()
    # 返回sheet的name ['structure', 'data', 'Sheet1', '备注', '命运卡', '地图（废弃）']
    print(sheet_names)
    # 将name中的内容 写入新文件的第0列就行了
    sheet = new_book.get_sheet(sheetId)
    for i in range(len(names)) :
        sheet.write(i, c, names[i])
    new_book.save('2.xls')

def getChar(n) :            # 将n映射成字符串
    # print('n = ' + str(n))
    charList = ["A", "B", "C", "D", "E", "F", "G", "H",
                "I", "J", "K", "L", "M", "N", "O", "P",
                "Q", "R", "S", "T", "U" ,"V", "W", "X", "Y", "Z"]
    if n == 0: return "A"
    ans = ""
    while n != 0:
        idx = int(n % 26)
        ans += charList[idx - 1]
        n = (int)((n - 1) / 26)
    ans = ans[::-1]
    return ans
def getNum(n) :
    if n == 0: return "0"
    ans = ""
    while n != 0:
        ans += str((n % 10))
        n = (int)(n /10)
    ans = ans[::-1]
    return ans

def change(s, n) :
    if s[-1].isdigit() :    # 如果是数字，直接加上字符
        s += getChar(n)
    else :
        s += getNum(n)      # 如果是字符，直接加上数字
    return s

def solve(list) :
    sorted(list)
    cnt = 1  # 重复的数字是现在第几名了
    for i in range(len(list) - 1):
        if (list[i] == list[i + 1]):
            list[i] = change(list[i], cnt)
            cnt += 1
        elif cnt != 1:  # 最后一个需要处理的
            list[i] = change(list[i], cnt)
            cnt = 1
    if cnt != 1:
        list[-1] = change(list[-1], cnt)
    return list

if __name__ == '__main__':

    print("要替换的文件需与脚本同一目录")
    fileName = input("输入文件名,加上扩展名字 ,比如1.xlsx:")
    sheetId = input("输入第几个sheet, 从1开始计数字:")
    sheetId = int(sheetId) - 1
    c = input("输入需要修改的是第几列，从1开始计数 :")
    c = int(c) - 1

    names = readEx(fileName, sheetId, c)
    names = solve(names)

    write(fileName,sheetId, names, c)
    print("修改好的列放在了2.xlsx")