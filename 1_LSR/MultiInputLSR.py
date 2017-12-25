import sys
import numpy
from numpy.lib import recfunctions
import datetime
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '../common')
import tools 


## WIP here, 
# Solutions would form a surface, which would be 2nd order, and I could do it at a later time but 
# at the moment I just want to get to other algorithms.

		# Because we have multiple sets of weights for the vectors now
		# Provide a container associating vectors with their weights.

class InputVector:
	def __init__(self, _vector, _weights, _alphas):
		self.weights = _weights
		self.vector = _vector
		self.alphas = _alphas

	weights = [] # Thetas for each term we are interested in, polynomials for now
	vector = [] # The input data for this vector
	alphas = [] # the alpha learning rate values associated with the polyweights for this vector

		# Note: Read InputVectors[weight][idx] 
		# as the input vector associated with weight "weight" at index idx (A single training input value)
		# Now generalized for multiple input vectors:
def batchGradientDescentStep(InputVectors, TargetVector):
	for vectorNum in range(len(InputVectors)):
		weights = InputVectors[vectorIdx].weights
		alphas = InputVectors[vectorIdx].alphas
		vector = InputVectors[vectorNum].vector
		for weightIdx in range(len(weights)):
			for vectorIdx in range(len(vector)):
				if len(Weights) == 2:
					sumOfDerivativeErrors += tools.firstOrderPolyErrors(weightIdx, idx, Weights, InputVectors, TargetVector)
				elif len(Weights) == 3:
					sumOfDerivativeErrors += tools.secondOrderPolyErrors(weightIdx, idx, Weights, InputVectors, TargetVector)

		print "	Subtracting: " + str(Alphas[weightIdx] * sumOfDerivativeErrors) + " From weight: " + str(weightIdx)
		Weights[weightIdx] -= InputVectors[weightIdx] * sumOfDerivativeErrors

	print "weights: " + str(Weights)
	return Weights

# This needs to be automated somehow . . .
dtypes = ('i4,i4,i4,i4,i4,f4,f4,f4,f4,f4,f4,f4')
usecolumns = (0,1,2,3,4,9,10,11,12,14,15,16)
fname = '../common/data/GuangzhouPM20100101_20151231.csv'

data = tools.loadAndPrintData(fname, dtypes, usecolumns)
data = tools.mergeTimeColumns(data, data["year"], data["month"], data["day"], data["hour"])

Input0 = 'HUMI'
Input1 = 'TEMP'
Target = 'precipitation'

dStart = 0
dEnd = 5200

data[Input0] = tools.interpolateBadNegatives(data[Input0])
#data[Input0] = data.normalizeVector(data[Input0])

data[Input1] = tools.interpolateBadNegatives(data[Input1])
#data[Input1] = data.normalizeVector(data[Input1])

data[Target] = tools.interpolateBadNegatives(data[Target])
data[Target] = tools.normalizeVector(data[Target])

Weights0 = [0.02, 0.00001] # for humi
Weights1 = [0.02, 0.00001] # for temp

alphas0 = [0.02, 0.00001] 
alphas1 = [0.02, 0.00001]

TargetVector = (data[Target][dStart:dEnd])
InputVectors = (InputVector(data[Input0][dStart:dEnd], Weights0, alphas0), InputVector(data[Input1][dStart:dEnd], Weights1, alphas1))



#####################################################
###### Animation functions and stuff ################
#####################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = InputVectors[0].vector
ys = InputVectors[1].vector
zs = TargetVector
print len(xs)
print len(ys)
print len(zs)

ax.scatter(xs, ys, zs, c='b', marker='o')

plt.show()


