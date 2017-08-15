# coding: utf-8
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])       # 2行4列的矩阵
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]              # shape是numpy中的一个函数，可以表示数组的维度。比如二维会显示行，列；[0]就表示行
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet         # tile（inX，i）表示按原来顺序拓展i个# tile(inX,(i,j))  i为拓展个数，j为拓展长度
    sqDiffMat = diffMat**2                      # 矩阵中每个元素的平方
    sqDistances = sqDiffMat.sum(axis=1)         # 对于二维数组，axis=0表示按列相加，1表示按行相加
    distances = sqDistances**0.5                # 矩阵中每个元素开方，计算距离
    sortedDistIndicies = distances.argsort()    # argsort()返回距离从小到大排序的索引值   a=（[4,5,1]）  返回【2，0，1】
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # dict里面给key赋value   选择距离最小的k个点
    sortedClassCount = sorted(classCount.iteritems(),               # iteritems 迭代输出字典的键值对排序
                              key=operator.itemgetter(1),           # key为函数，指定取待排序元素的哪一项进行排序
                              reverse=True)                         # reverse是一个bool变量，默认为false（升序排列），定义为True时将按降序排列。
    return sortedClassCount[0][0]


def file2matrix(filename):                      # 使用Matplotlib创建散点图
    fr = open(filename)                         # 打开文件
    arrayOLines = fr.readlines()                # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
    numberOfLines = len(arrayOLines)            # 返回长度
    returnMat = zeros((numberOfLines, 3))       # 创建给定类型的矩阵，并初始化为0
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()                     # strip() 方法用于移除字符串头尾指定的字符（默认删除空白符（包括'\n', '\r',  '\t',  ' ')）
        listFromLine = line.split('\t')         # split()通过指定分隔符对字符串进行切片，如果参数num 有指定值，则仅分隔 num 个子字符串
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))  # python中可以使用索引值-1表示列表中的最后一列元素。告诉它列表存储的元素值为整型，否则python语言会当做字符串处理
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)                    # 参数 0使得函数可以从列中选取最小值，而不是选取当 前行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))         # shape 参数若为矩阵，则返回行列的维数
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))  #这里的/是具体特征值相除，numpy中矩阵除法需要函数
    return  normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10                              # 测试数据占得比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],normMat[numTestVecs:m, :],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" %(errorCount/float(numTestVecs))


def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels, = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels,3)
    print "You will probably like this person: ", resultList[classifierResult - 1]


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect




























