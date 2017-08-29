# coding:utf-8
def loaddataset():
    postinglist = [['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classvec = [0, 1, 0, 1, 0, 1]
    return postinglist, classvec


def createvocablist(dataset):                                   # 创建一个包含在所有文档中出现的不重复词的列表
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set(document)                     # |用于求两个集合的并集
    return list(vocabset)


def setofwords2vec(vocablist, inputset):
    returnvec = [0]*len(vocablist)                              # 创建一个和词汇表等长的向量，元素置0
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1                    # 遍历文档中所有单词，如果出现词汇表中的，置1d
        else:
            print "the word: %s is mot in my Vocabulary" % word
    return returnvec
