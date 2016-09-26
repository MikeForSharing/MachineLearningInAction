from math import log

def calcShannonEnt(dataSet):
    numSample = len(dataSet)
    labelsCount = {}
    for firstDim in dataSet:
        currentLabel = firstDim[-1]
        if currentLabel not in labelsCount:
            labelsCount[currentLabel] = 0
        labelsCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelsCount:
        prob = float(labelsCount[key])/numSample
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[0,1,'no'],[1,0,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels
