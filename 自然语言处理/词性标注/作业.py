# 1、利用词性标注方法对新闻文本（481676.txt）分词后先标注词性，再识别其中的人名、地名、机构名、日期等命名实体；
import jieba.posseg as ps


# 读取文本信息
def readFile(path):
    with open(path, 'r', encoding="utf-8") as file:
        str_doc = file.read()
    return str_doc


# 自定义特征抽取规则
def deal_words(str_doc):
    findwords = ""
    stwlist = get_stop_words()
    user_pos_list = ['nr', 'ns', 'nt', 't']
    for word, pos in ps.cut(str_doc):
        if word not in stwlist and pos in user_pos_list:
            if word + ' ' + pos + '\n' not in findwords:
                findwords += word + ' ' + pos + '\n'
    print(findwords)


# 创建去停用词的列表
def get_stop_words():
    file = open('dropwords.txt', 'r', encoding='utf-8').read().split('\n')
    return set(file)


str_doc = readFile("481652.txt")
deal_words(str_doc)
