import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add__subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
import KNN
datingDataMat,datingLabels = KNN.file2matrix("datingTestSet.txt")
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
help(array())
from numpy import array

norMat, range, minValue = KNN.autoNorm(datingDataMat)