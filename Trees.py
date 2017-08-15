# coding=utf-8
import operator
from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)                           #实例总数
    labelCounts = {}                                    #s数据字典，它的键值是最后一列数值
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():      #如果当前键值不存在，则扩展字典将当前键值加入字典
            labelCounts[currentLabel] = 0                   #每个键值都记录了当前类别出现的次数
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries       #计算概率
        shannonEnt -= prob * log(prob,2)                #香农熵计算公式
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):                 #划分数据集
    retDataSet = []                                     #创建新的list，保护原始数据集
    for featVec in dataSet:                             #遍历dataSet
        if featVec[axis] == value:                      #一旦发现符合要求的值，将其添加到新建的list中
            reducedFeatVec = featVec[: axis]            #切片
            reducedFeatVec.extend(featVec[axis+1:])     #extend 将参数的所有元素加到list
            retDataSet.append(reducedFeatVec)           #append 将参数作为一个元素加到list
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)               #计算整个数据集的原始香农熵
    bestInfoGain = 0.0; bestFeatures = -1
    for i in range(numFeatures):                        #遍历数据集中所有特征
        featList = [example[i] for example in dataSet]  #列表生成式，将数据集中所有第i个特征值或所有可能存在的值写入新list
        uniqueVals = set(featList)                      #set 集合数据类型，和list类似，但元素互不相同， 是得到list中唯一元素值最快的方法
        newEntropy = 0.0
        for value in uniqueVals:                        #对每个特征划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy             #信息增益
        if(infoGain > bestInfoGain):                    #比较所有特征中的信息增益
            bestInfoGain = infoGain
            bestFeatures = i
        return bestFeatures

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                            key = operator.itemgetter(1),
                            reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
        if len(dataSet[0]) == 1:
            return majorityCnt(classList)
        bestFeat = chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        del(labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        return myTree
    
