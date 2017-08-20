# coding=utf-8
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")                  # 定义文本框和箭头格式
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):                   # 绘制带箭头的注解
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
    xytext=centerPt, textcoords='axes fraction', va="center", ha="center",
    bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getnumleafs(myTree):                                            # 叶子节点数目
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                  # 节点的数据类型是否为字典
            numLeafs += getnumleafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def gettreedepth(myTree):                                           # 层数
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + gettreedepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrievetree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers':
                    {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers':
                    {0: {'heda': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


def plotmidtext(cntrPt, parentPt, txtString):                           # 在父子节点间填充文本信息
    xmid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    ymid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xmid, ymid, txtString)


def plottree(myTree, parentPt, nodeTxt):
    numLeafs = getnumleafs(myTree)
    depth = gettreedepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plottree.xOff + (1.0 + float(numLeafs))/2.0/plottree.totalW, plottree.yOff)
    plotmidtext(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plottree.yOff = plottree.yOff - 1.0/plottree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plottree(secondDict[key], cntrPt, str(key))
        else:
            plottree.xOff = plottree.xOff + 1.0/plottree.totalW
            plotNode(secondDict[key], (plottree.xOff, plottree.yOff), cntrPt, leafNode)
            plotmidtext((plottree.xOff, plottree.yOff), cntrPt, str(key))
    plottree.yOff = plottree.yOff + 1.0/plottree.totalW


def createplot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plottree.totalW = float(getnumleafs(intree))                    # 树的宽度
    plottree.totalD = float(gettreedepth(intree))                   # 树的深度
    plottree.xOff = -0.5/plottree.totalW
    plottree.yOff = 1.0
    plottree(intree, (0.5, 1.0), '')
    plt.show()





























