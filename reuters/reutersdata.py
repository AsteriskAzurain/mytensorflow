import tensorflow as tf
from tensorflow.keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # 下标-单词对应字典

decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
# 偏移3个：0,1,2保留下标，分别表示：“padding,” “start of sequence,” and “unknown.”
'''
原文
'? ? ? said as a result of its december acquisition of space co it 
expects earnings per share in 1987 of 1 15 to 1 30 dlrs per share up from 70 cts 
in 1986 the company said pretax net should rise to nine to 10 mln dlrs from six 
mln dlrs in 1986 and rental operation revenues to 19 to 22 mln dlrs from 12 5 
mln dlrs it said cash flow per share this year should be 2 50 to three dlrs reuter 3'
'''