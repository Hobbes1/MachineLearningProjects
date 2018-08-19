#GDAClassifier.py
import numpy as np
from collections import namedtuple
from scipy.stats import multivariate_normal

class GDAClassifier(object):

	def __init__(self, testNumA, testNumB, pointsA, pointsB):
		'''
		Constructor
		@param testNumA: 	The number of points used to generate the function distA
		@param testNuMB: 	The number of points used to generate the function distB
		@param pointsA: 	numpy array of points of class A
		@param pointsB:		numpy array of points of class B
		'''
		self.testNumA = testNumA
		self.testNumB = testNumB
		self.pointsA = pointsA
		self.pointsB = pointsB		


	def classifyPoints(self):
		'''
		Generate gaussian PDF's for self.pointsA and self.pointsB making use of 
		self.testNumA points from pointsA, and self.testNumB points from pointsB
		Assigns self.gaussDistA and self.gaussDistB.

		Then make predictions on the rest of the points using those distributions
		@returns results: 		A named tuple with members:
								results.numCorrectA,
								results.numCorrectB,
								results.numIncorrectA,
								results.numIncorrectB
		'''

		maxMuAx = np.average(self.pointsA[0:self.testNumA,0])
		maxMuAy = np.average(self.pointsA[0:self.testNumA,1])
		maxMuBx = np.average(self.pointsB[0:self.testNumB,0])
		maxMuBy = np.average(self.pointsB[0:self.testNumB,1])
		estCovarianceA = np.cov(self.pointsA[0:self.testNumA].T)
		estCovarianceB = np.cov(self.pointsB[0:self.testNumB].T)
		self.gaussDistA = multivariate_normal([maxMuAx, maxMuAy], estCovarianceA)
		self.gaussDistB = multivariate_normal([maxMuBx, maxMuBy], estCovarianceB)

		numCorrectA = 0
		numIncorrectA = 0
		numCorrectB = 0
		numIncorrectB = 0
		classA = 0
		classB = 1
		for i in range(self.testNumA, self.pointsA.shape[0]):
			res = self._predict(self.pointsA[i], classA)
			if res == 1:
				numCorrectA += 1
			else:
				numIncorrectA += 1

		for i in range(self.testNumB, self.pointsB.shape[0]):
			res = self._predict(self.pointsB[i], classB)
			if res == 1:
				numCorrectB += 1
			else:
				numIncorrectB += 1

		result = namedtuple("results", ["numCorrectA", "numCorrectB", "numIncorrectA", "numIncorrectB", "gaussDistA", "gaussDistB"])
		return result(numCorrectA, numCorrectB, numIncorrectA, numIncorrectB, self.gaussDistA, self.gaussDistB)


	def _predict(self, point, actualClass):
		''' 
		Make the prediction of the class of the point given the point evaluated at the PDF's A and B
		using Bayes rule predictions: P(Y|x) = (P(X|y)*P(y)/(P(x))). Intended for use with only the known data, 
		the points that were used to calculate distA and distB, to predict the values of the remaining points.
		Return 1 if the actualClass(a number, 0:A, 1:B) is equal to the predicted class.
		Return 0 otherwise

		@param point: 		The point, a list-like with 2 members for x and y values
		@param actualClass:	Either 0 or 1, as described above
		'''
		pA = (self.testNumA / (self.testNumA + self.testNumB))
		pB = (self.testNumB / (self.testNumA + self.testNumB))
		pXgivenA = self.gaussDistA.pdf(point) 
		pXgivenB = self.gaussDistB.pdf(point)

		# This term IS unecessary, the argmax evaluation does not 
		# change as it is common between pAgivenX and pBgivenX
		pX = pXgivenA * pA + pXgivenB * pB

		pAgivenX = pXgivenA * pA / pX
		pBgivenX = pXgivenB * pB / pX

		if (pAgivenX > pBgivenX):
			if actualClass == 0:
				return 1
			else:
				return 0
		else:
			if actualClass == 1:
				return 1
			else:
				return 0
