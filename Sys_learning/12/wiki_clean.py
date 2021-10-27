#encoding:utf8
import jieba
import os
import codecs
from tqdm import tqdm


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                if len(line) > 0:
                    yield [segment.strip() for segment in jieba.cut(line.strip(), cut_all=False)
                           if segment not in stoplist and len(segment) > 0]


def is_ustr(instr):
    out_str = ''
    for index in range(len(instr)):
        if is_uchar(instr[index]):
            out_str = out_str + instr[index].strip()
    return out_str


def is_uchar(uchar):
    # """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fff':
        return True

if __name__ == '__main__':
    dirname = 'zh_simplify'
    # 读取停用词；
    stop_f = codecs.open(u'停用词.txt', 'r', encoding='utf-8')
    stoplist = {}.fromkeys([line.strip() for line in stop_f])
    # 进行jieba分词
    sentences = MySentences(dirname)
    # 分词结果写入文件
    f = codecs.open('wiki_jieba.txt', 'w', encoding='utf-8')
    i = 0
    j = 0
    w = tqdm(sentences, desc=u'分词句子')
    for sentence in w:
        if len(sentence) > 0:
            output = " "
            for d in sentence:
                # 去除停用词；
                if d not in stoplist:
                    output += is_ustr(d).strip() + " "
            f.write(output.strip())
            f.write('\r\n')
            i += 1
            if i % 10000 == 0:
                j += 1
                w.set_description(u'已分词： %s万个句子'%j)
    f.close()