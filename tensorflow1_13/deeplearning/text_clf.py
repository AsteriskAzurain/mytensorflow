from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
vectorizer = CountVectorizer()
transformer = TfidfTransformer()

text_tr = [
    'My dog has flea problems, help please.',
    'Maybe not take him to dog park is stupid.',
    'My dalmation is so cute. I love him my.',
    'Stop posting stupid worthless garbage.'
]
text_te = [
    'Mr licks ate mu steak, what can I do?.',
    'Quit buying worthless dog food stupid'
]
y_tr = [0, 1, 0, 1]
y_te = [0, 1]

count_tr = vectorizer.fit_transform(text_tr).toarray()       # 转成词向量
tfidf_tr = transformer.fit_transform(count_tr).toarray()     # 转成tf-idf权值

count_te = CountVectorizer(vocabulary=vectorizer.vocabulary_).fit_transform(text_te).toarray()   # 转成词向量
tfidf_te = transformer.fit_transform(count_te).toarray()  # 转成tf-idf权值

model = GaussianNB()
model.fit(tfidf_tr, y_tr)    # 模型训练
model.predict(tfidf_te)      # 模型预测

