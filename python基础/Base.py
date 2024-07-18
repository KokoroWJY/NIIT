def Xunhuan():
    # for 循环
    for i in range(10):  # 循环控制表达式
        print(i, end="\t")
        i += 1  # 循环控制变量的增量
    else:
        print()
        print("结束了! i = ", i)


def Test():
    for i in range(1, 6):
        for r in range(i):
            print("*", end='\t')
        print()


if __name__ == '__main__':
    Test()
