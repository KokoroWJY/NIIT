# 分析新闻文本（481652.txt），经过分词和去停用词后，实现词频统计、可视化，自定义去高低频词。
import jieba
import matplotlib
from nltk import *

matplotlib.rcParams['font.sans-serif'] = 'SimHei'
with open('481652.txt', 'r+', encoding='utf-8') as file:
    data = file.read()
    # 分词
    seg_list = jieba.cut(data, cut_all=False)
    # new_seg_list=list(seg_list)
    # print(new_seg_list)
    seg_list1 = jieba.cut(data, cut_all=False)
    words = '/'.join(seg_list1)
    print('分词：' + words)


def stopwordslist():
    stopwords = [line.strip() for line in open('dropwords.txt', encoding='UTF-8').readlines()]
    return stopwords


split_words = [x for x in seg_list if x not in stopwordslist()]
new_str = '/'.join(split_words)
print('停用词：', new_str)
# 实现词频统计
fdist = FreqDist(split_words)
print('词频统计：', fdist.keys(), '\n', fdist.values(), '\n')
# 可视化
fdist.plot(20)


# 自定义去高低频词。
def freqword(fdist, mincount, maxcount):
    wordlist = []
    for key in fdist.keys():
        if fdist.get(key) > mincount and fdist.get(key) < maxcount:
            wordlist.append(key + ':' + str(fdist.get(key)))
    return wordlist


new_list = freqword(fdist, 3, 10)
print('自定义去高低频词:', new_list)
