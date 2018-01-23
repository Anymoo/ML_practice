import numpy as np
import matplotlib.pyplot as plt
import random
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg


def loadDataSet():
    dataMat = []
    lableMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        lableMat.append(int(lineArr[2]))
    return dataMat, lableMat


def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))


def gradAscent(dataMatIn, classLables):
    dataMatrix = np.mat(dataMatIn)
    lableMat = np.mat(classLables).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (lableMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


#画决策边界
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat, lableMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(lableMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()




np.random.seed(1234)

# 加载数据
def loaddata():
    dataMat = []
    labelMat = []
    for line in open('testSet.txt', 'r'):
        line = line.strip().split()
        dataMat.append([1.0, float(line[0]), float(line[1])])
        labelMat.append(int(line[2]))
    return dataMat, labelMat

# 显示散点图
def plotDataSet():
    data, label = loaddata()
    data = np.array(data)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(data.shape[0]):
        if int(label[i]) == 1:
            xcord1.append(data[i, 1])
            ycord1.append(data[i, 2])
        else:
            xcord2.append(data[i, 1])
            ycord2.append(data[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker='s', alpha = 0.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', alpha = 0.5)
    plt.title('DataSet')
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.show()


#随机梯度上升
def stocGradAscent0(dataMatrix, classLables):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLables[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


#改进的随机梯度上升
def stocGradAscent1(dataMatrix, classLables,numIter=150):
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLables[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


#预测病马
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount +=1
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate of this test is: %f'% errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' % (numTests, errorSum/float(numTests)))
