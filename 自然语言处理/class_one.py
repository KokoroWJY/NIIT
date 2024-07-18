import nltk

sentence = 'FIFA was founded in 1904 to oversee international competition among the national associations of Belgium, '
# 对句子进行分词
tokens = nltk.word_tokenize(sentence)
print('tokens:\n', tokens)

# 对分完词的句子进行词性标注
tagged = nltk.pos_tag(tokens)
print('tagged:\n', tagged)

# 对标注词性的词进行命名实体识别
entities = nltk.chunk.ne_chunk(tagged)
# 以NNP为结尾的是实体，GPE通常表示地理-政治条目
print('entities:\n', entities)
print('=' * 50)
# 筛选实体
for x in str(entities).split('\n'):
    if '/NNP' in x:
        print(x)
