import re

regex = '.*高考的时间是(\d{4}[年/-]\d{1,2}[月/-]?\d{0,2}[日/-]?)'
string1 = 'XXX高考的时间是2018年6月7日'
re_string1 = re.search(regex, string1)
print(re_string1.group(1))

string2 = 'XXX高考的时间是2018/6/7'
re_string2 = re.search(regex, string2)
print(re_string2.group(1))

string3 = 'XXX高考的时间是2018-6-7'
re_string3 = re.search(regex, string3)
print(re_string3.group(1))

string4 = 'XXX高考的时间是2018-06-07'
re_string4 = re.search(regex, string4)
print(re_string4.group(1))

string5 = 'XXX高考的时间是2018-06'
re_string5 = re.search(regex, string5)
print(re_string5.group(1))

string6 = 'XXX高考的时间是2018年6月'
re_string6 = re.search(regex, string6)
print(re_string6.group(1))



# 对新闻文本（481651.txt）清除文本中的特殊符号、标点、英文、数字等，仅保留汉字信息；同时去除换行符，将多个空格变成一个空格。
import re

str1 = ''
with open('481651.txt', 'r+', encoding='utf-8') as file:
    text = file.read()
    # print(text)
    text = re.sub(r'\n', '', text)  # 去除换行符
    # print(text)

    text = re.sub(r'\s{2,}', ' ', text)  # 将多个空格变成一个空格
    # print(text)

    text = re.sub('[a-zA-Z0-9]', '', text)  # 清除英文、数字
    # print(text)

    text = re.sub(r'[^\u4e00-\u9fa5\s]', '', text)  # 仅保留汉字信息和空格
    str1 = str1 + text
    print(text)
    file.close()
# print(str1)
with open('481651.txt', 'w+', encoding='utf-8') as file:
    file.write(str1)
    file.close()
