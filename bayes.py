# coding:utf-8
from numpy import *
import math

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


def trainnb0(trainmatrix, traincategory):
    numtraindocs = len(trainmatrix)
    numwords = len(trainmatrix[0])
    pabusive = sum(traincategory)/float(numtraindocs)
    p0num = ones(numwords)                              # 将所有词的出现数初始化为1
    p1num = ones(numwords)
    p0denom = 2.0                                       # 分母初始化为2
    p1denom = 2.0
    for i in range(numtraindocs):                       # 对每列做以下操作
        if traincategory[i] == 1:                       # 标记为1的那一列
            p1num += trainmatrix[i]                     # 标记为1的那列数值相加
            p1denom += sum(trainmatrix[i])              # 标记为1的那列元素的个数
        else:
            p0num += trainmatrix[i]
            p0denom += sum(trainmatrix[i])
    p1vect = math.log(p1num/p1denom)                         # 避免下溢，使用对数
    p0vect = math.log(p0num/p0denom)
    return p0vect, p1vect, pabusive


def classifynb(vec2classify, p0vec, p1vec, pclass1):
    p1 = sum(vec2classify * p1vec) + logspace(pclass1)
    p0 = sum(vec2classify * p0vec) + logspace(1.0 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0
    