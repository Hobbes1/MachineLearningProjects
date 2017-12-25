import sys
import numpy
from numpy.lib import recfunctions
import datetime
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines

sys.path.insert(0, '../common')
import tools 

		# Note: Read InputVectors[weight][idx] 
		# as the input vector associated with weight "weight" at index idx (A single training input value)

def batchGradientDescentStep(Weights, InputVectors, TargetVector, Alphas):
	for weightIdx in range(len(Weights)):
		sumOfDerivativeErrors = 0.0
		for idx in range(len(InputVectors[0])):
				# Subtract from the weight the gradient in the associated Input direction
				#w hose derivative is equal to the sum [ Err (not squared) * X_i ] where
				# X_i is the InputVector associated with the weight.

			if len(Weights) == 2:
				sumOfDerivativeErrors = tools.firstOrderPolyErrors(weightIdx, idx, Weights, InputVectors, TargetVector)
			elif len(Weights) == 3:
				sumOfDerivativeErrors = tools.secondOrderPolyErrors(weightIdx, idx, Weights, InputVectors, TargetVector)
			#Theta_0, the constant factor
			'''
			if weightIdx == 0:
				sumOfDerivativeErrors += ((Weights[0] + Weights[1] * InputVectors[weightIdx][idx] - TargetVector[idx]))
			#Theta_1, the linear factor
			elif weightIdx == 1:
				sumOfDerivativeErrors += (Weights[0] + Weights[1] * InputVectors[weightIdx][idx] - TargetVector[idx]) * InputVectors[weightIdx][idx]
			elif (len(Weights>2)):
				if weightIdx == 2:
					sumOfDerivativeErrors += (Weights[0] + Weights[1] * InputVectors[])
			'''
			# Doing so for all weights constitudes one step of gradient descent 
			# Higher order polynomial derivative terms would go here
			# or any guess term (Sigmoid, hyperbolic eqn, (fourier terms?))

		print "	Subtracting: " + str(Alphas[weightIdx] * sumOfDerivativeErrors) + " From weight: " + str(weightIdx)
		Weights[weightIdx] -= Alphas[weightIdx] * sumOfDerivativeErrors

	print "weights: " + str(Weights)
	return Weights


##############################
###### actual code ###########
##############################

# This needs to be automated somehow . . .
dtypes = ('i4,i4,i4,i4,i4,f4,f4,f4,f4,f4,f4,f4')
usecolumns = (0,1,2,3,4,9,10,11,12,14,15,16)
fname = '../common/data/GuangzhouPM20100101_20151231.csv'

data = tools.loadAndPrintData(fname, dtypes, usecolumns)
data = tools.mergeTimeColumns(data, data["year"], data["month"], data["day"], data["hour"])

Input = 'hours'
Target = 'TEMP'
dStart = 0
dEnd = 9000
data[Target] = tools.interpolateBadNegatives(data[Target])
data[Target] = tools.normalizeVector(data[Target])
print data[Target]
data[Input] = tools.interpolateBadNegatives(data[Input])


NumSamples = data.size
TargetVector = (data[Target][dStart:dEnd])
		# I make weights and inputs vectors, or vectors of vectors 
		# respectively regardless of there being one or many, so that I can call
		# one set of functions on them making use of range() regardless
InputVectors = [data[Input][dStart:dEnd]]
Weights = [0.000015, 0.00000001, 0.0];

# For the WP, it took me the most time to figure out a few things:
# - Shit wasn't working because I had bad data that I hadn't visualized (negatives)
# - Regression rates for different polynomial terms should be different, and
# - The rate for learning the constant factor needed to be far higher than the linear factor,
#   otherwise it appeared the regression got stuck in a local minimum caused by the linear
#   term reaching an equilibrium before the constant term got a chance to move.

#ratio of rates: 2 : 0.00000001
#rates for 2nd order: [0.018, 0.0000000005,0.000000000000000025]
Alphas = [0.018, 0.0000000005,0.000000000000000025]

print TargetVector.shape
print TargetVector
print InputVectors
print Weights 

#####################################################
###### Animation functions and stuff ################
#####################################################

plt.figure(figsize=(8,6))

plt.scatter(data[Input][dStart:dEnd], data[Target][dStart:dEnd], s=1)
plt.axis([0, numpy.amax(data[Input][dStart:dEnd]), 0, numpy.amax(data[Target][dStart:dEnd])])

plt.gcf().set_facecolor((0.2, 0.2, 0.22))
plt.gcf().figsize = (8, 6)
ax = plt.subplot()
ax.set_facecolor((0.2,0.2,0.22))
ax.set_xlabel(Input, color=(0.8,0.82,0.82), size=16)
ax.set_ylabel(Target, color=(0.8,0.82,0.82), size=16)
ax.spines['bottom'].set_color((0.8,0.82,0.82))
ax.spines['left'].set_color((0.8,0.82,0.82))

ax.spines['top'].set_color((0.2,0.2,0.22))
ax.spines['right'].set_color((0.2,0.2,0.22))

ax.tick_params(axis='x', colors=(0.8,0.82,0.82), size=3)
ax.tick_params(axis='y', colors=(0.8,0.82,0.82), size=3)

err = 0
steps = 120
pad = len(str(steps))
f = 0

record = False
if sys.argv[1] == 'record':
	record = True

xs = numpy.arange(0, dEnd, (dEnd - dStart)/100)
ys = numpy.zeros(100)

myLines = []
for i in range(steps):
	Weights = batchGradientDescentStep(Weights, InputVectors, TargetVector, Alphas)
	err = tools.squareError(Weights, InputVectors, TargetVector)
	print "	Weights: " + str(Weights) 
	print "	Err: " + str(err)

	if (len(Weights) == 2):
		myLines.append(lines.Line2D([0, numpy.amax(data[Input])], [Weights[0], Weights[0] + Weights[1]*numpy.amax(data[Input])], color='r'))
		ax.add_line(myLines[i])
	else:
		ys = tools.formSecondOrderPoly(xs, Weights)
		plt.plot(xs, ys, color='r')

	if (record):	
		if i > 2:
			ax.lines[0].remove()
			ax.lines[0].set_alpha(0.1)
			ax.lines[1].set_alpha(0.5)

		name = '../images/frame_'+str(f).zfill(pad)+'.png'
		f+=1
		plt.savefig(name, bbox_inches='tight', facecolor=(0.2,0.2,0.22))

plt.show()