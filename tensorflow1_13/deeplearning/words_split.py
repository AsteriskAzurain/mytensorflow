import jieba

string = '大家好，我是张敏，我正在学习NLP课程。'

res = jieba.lcut(string)
list(res)

jieba.add_word('大家好')   # 添加一个词语进入词典
jieba.load_userdict('def_word.txt')  # 批量添加词语进词典

