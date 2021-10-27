
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def clean(text):
    #     remove urls
    text = re.sub(r'http\S+', " ", text)
    #     remove mentions
    text = re.sub(r'@\w+', ' ', text)
    #     remove hastags
    text = re.sub(r'#\w+', ' ', text)
    #     remove digits
    text = re.sub(r'\d+', ' ', text)
    #     remove html tags
    text = re.sub('r<.*?>', ' ', text)
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    r3 =  "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
    r4 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    #text = re.sub(r4,'',text)
    #     remove stop words
    text = text.split()
    text = " ".join([word.lower() for word in text if not word in stop_word])

    return text


def convert(sentiment):
    if sentiment == 'Positive' or sentiment == 'Extremely Positive':
        return '1'
    elif sentiment == 'Negative' or sentiment == 'Extremely Negative':
        return '-1'
    elif sentiment == 'Neutral':
        return '0'


if __name__ == '__main__':
    train = pd.read_csv("../input/covid-19-nlp-text-classification/Corona_NLP_train.csv")
    test = pd.read_csv("../input/covid-19-nlp-text-classification/Corona_NLP_test.csv")
    stop_word = stopwords.words('english')
    train['OriginalTweet'] = train['OriginalTweet'].apply(lambda x: clean(x))
    test['OriginalTweet'] = test['OriginalTweet'].apply(lambda x: clean(x))
    train['Sentiment'] = train['Sentiment'].apply(lambda x: convert(x))
    test['Sentiment'] = test['Sentiment'].apply(lambda x: convert(x))
    train = train.iloc[:, 4:]
    test = test.iloc[:, 4:]

    labels = []
    contents = []
    for i in train['OriginalTweet']:
        contents.append(i)
    for i in train['Sentiment']:
        labels.append(i)
    labels_te = []
    for i in test['OriginalTweet']:
        contents.append(i)
    for i in test['Sentiment']:
        labels_te.append(i)

        # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(contents))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    # X = tfidf.toarray()
    X = coo_matrix(tfidf, dtype=np.float32).toarray()  # 稀疏矩阵 注意float

    X_train, X_test = X[:41157], X[41157:]
    y_train, y_test = labels, labels_te

    LR = LogisticRegression(solver='liblinear')
    LR.fit(X_train, y_train)
    print('模型的准确度:{}'.format(LR.score(X_test, y_test)))
    pre = LR.predict(X_test)
    print("逻辑回归分类")
    print(len(pre), len(y_test))
    print(classification_report(y_test, pre))
    print("\n")
