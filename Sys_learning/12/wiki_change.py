#encoding:utf8
import codecs
from tqdm import tqdm

#  简繁转化


def transformFile(ipath, opath):
    encoding = 'utf-16-le'
    iFile = codecs.open(ipath, 'r', encoding)
    encoding = 'utf-8'
    oFile = codecs.open(opath, 'w', encoding)
    sentences = iFile.readlines()
    i = 0
    w = tqdm(sentences, desc=u'转换句子')
    for sentence in w:
        oFile.write(sentence)
        i += 1
        if i % 100 == 0:
            w.set_description(u'已转换%s个句子'%i)
    iFile.close()
    oFile.close()

ipath = 'wiki.zh.jian.txt'
opath = 'wiki.zh.jian.utf8.txt'
transformFile(ipath, opath)


