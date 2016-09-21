from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels	




def classify0(inItem, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inItem, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
		sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
		return sortedClassCount[0][0]


def file2matrix(fileName):
	file = open(fileName)
	arrayOfFile = file.readlines()
	numberOfLines = len(arrayOfFile)
	returnMat = zeros((numberOfLines,3))
	classLabels = []
	index = 0
	for line in arrayOfFile:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabels.append(listFromLine[-1])
		index += 1

	return returnMat, classLabels


def autoNorm(dataSet):
	maxValue = dataSet.max(0)
	minValue = dataSet.min(0)
	rangeValue = maxValue - minValue
	NorMat = zeros(shape(dataSet))
	m = dataSet.shape[0]
	NorMat = dataSet - tile(minValue,(m,1))
	NorMat = NorMat/tile(rangeValue, (m,1) )
	return NorMat, rangeValue, minValue


def datingClassTest():
	hoRatio = 0.1
	datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
	NorMat, rangeVal, minVal = autoNorm(datingDataMat)
	m = NorMat.shape[0]
	NumTestVecs = int(m * hoRatio)
	errorCount = 0.0
	for i in range(NumTestVecs):
		classifierResult = classify0(NorMat[i,:],NorMat[NumTestVecs:m,:],datingLabels[NumTestVecs:m],3)
		print "the classifier come back the result %s, the real result is %s" %(classifierResult,datingLabels[i])
		if(classifierResult != datingLabels[i]):
			errorCount += 1.0
	print "the total error rate is %f" %(errorCount/float(NumTestVecs))



def classifyPerson():
	percentGames = float(raw_input("please input the percentage of time to play games"))
	ffMiles = float(raw_input("please input the frequent fly miles earned per year"))
	iceLitre = float(raw_input("please input the litre consumed per year"))

	datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
	norMat, rangeVal, minVal = autoNorm(datingDataMat)
	itemArr = [ffMiles, percentGames, iceLitre]
	classifyResult = classify0((itemArr-minVal)/rangeVal, norMat, datingLabels, 3)
	print "the level you will like this persion is %s" %(classifyResult)