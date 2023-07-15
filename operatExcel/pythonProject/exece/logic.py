"""

操作的逻辑

命名采用26进制
满26进A

1. 如果排好了顺序
    1. i表示第几次循环
    2，j表示当前的字母应该是多少
"""

# list = ['A', 'B','C','D','E']
# for i in range(1<<5) :
#     for j in range(5):
#         if i>> j & 1:
#             print(list[j], end='')
#     print()


list1 = ["fda"]
list2 = [ "123", "123", "123"]
# list = list1 * 11
list = list2 * 100
# print((int) (26 % 26))

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

cnt = 1         # 重复的数字是现在第几名了
for i in range(len(list) - 1) :
    if (list[i] == list[i + 1]) :
        list[i] = change(list[i], cnt)
        cnt += 1
    elif cnt != 0 :         # 最后一个需要处理的
        list[i] = change(list[i], cnt)
        cnt = 0
if cnt != 0:
    list[-1] = change(list[-1], cnt)

for s in list :
    print(s)
