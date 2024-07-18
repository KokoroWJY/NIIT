import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文支持
import matplotlib as mpl

mpl.rcParams["font.family"] = "Kaiti"  # SimHei 中文黑体 Kaiti 中文楷体
mpl.rcParams["axes.unicode_minus"] = False  # 显示负数

df = pd.read_csv()
ls = np.array([1, 2, 3])

'''正则表达式
        1. match匹配
            1.1 匹配字符串'''
import re

rst = re.match('www', 'www.baidu.com')

