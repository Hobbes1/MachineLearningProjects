import sys
import numpy
from numpy.lib import recfunctions
import datetime
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

sys.path.insert(0, '../common')
import tools 

#	This could work if one spent the time to find the proper alpha 
#	values for every local portion of their regression . . .
#	I don't see how that could ever be desireable so I'm not going to 
# 	bother now that I understand the concept and practice.

# Wonderful, now theres local weights and function weights
def stochasticLWGradientDescentStep(Weights, localWeights, InputVectors, TargetVector, Alphas, idx):

	for weightIdx in range(len(Weights)):
		sumOfDerivativeErrors = 0.0
		if len(Weights) == 2:
			sumOfDerivativeErrors += localWeights[idx] * tools.firstOrderPolyErrors(weightIdx, idx, Weights, InputVectors, TargetVector)
			
		elif len(Weights) == 3:
			sumOfDerivativeErrors += localWeights[idx] * tools.secondOrderPolyErrors(weightIdx, idx, Weights, InputVectors, TargetVector)

		#print 'subtracting: ' + str(Alphas[weightIdx] * sumOfDerivativeErrors) + ' from weight ' + str(weightIdx) + ": " + str(Weights[weightIdx])
		Weights[weightIdx] -= Alphas[weightIdx] * sumOfDerivativeErrors

	return Weights

localWeights = [] 

# This needs to be automated somehow . . .
dtypes = ('i4,i4,i4,i4,i4,f4,f4,f4,f4,f4,f4,f4')
usecolumns = (0,1,2,3,4,9,10,11,12,14,15,16)
fname = '../common/data/GuangzhouPM20100101_20151231.csv'

data = tools.loadAndPrintData(fname, dtypes, usecolumns)
data = tools.mergeTimeColumns(data, data["year"], data["month"], data["day"], data["hour"])

Input = 'hours'
Target = 'TEMP'

data[Target] = tools.interpolateBadNegatives(data[Target])
data[Target] = tools.normalizeVector(data[Target])
print data[Target]
data[Input] = tools.interpolateBadNegatives(data[Input])

dStart = 0
dEnd = 4000

localSteps = 1000
localSampleNum = dEnd/localSteps

TargetVector = (data[Target][dStart:dEnd])
InputVectors = [data[Input][dStart:dEnd]]
Weights = [0.0, 0.0]
Alphas = [0.0002, 0.00000003]

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


record = False
if sys.argv[1] == 'record':
	record = True
pad = 2
f = 0
InputVectorsReal = []
trainSteps = 30
myLines = []
colors = ['r', 'g', 'k', 'w', 'y', 'o', 'v']
sums = [0, 0, 0, 0, 0, 0, 0]
idx = 0

# Could functionalize this in the future, output mapped error surface given
# the space of two weight choices (here, constant offset and linear term)

# InputVector = data[Input][300:450]
# TargetVector = data[Target][300:450]
# size = 100
# startx = -500.0
# endx = 500.0
# starty= -5e-1
# endy = 5e-1
# zs = tools.genErrorSpaceTwoVars(startx, endx, starty, endy, size + 1, InputVector, TargetVector)

# xs = numpy.zeros([size,size])
# ys = numpy.zeros([size,size])
# for xIdx in range((size)):
# 	for yIdx in range((size)):
# 		xs[xIdx][yIdx] = (startx - endx) / size * xIdx + -5e-5
# 		ys[xIdx][yIdx] = (starty - endy) / size * yIdx + -5e-8

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm)
# plt.show()

for i in range(localSampleNum):
	# Always regress as if your data started at zero to prevent lienar error terms from regressing at faster
	# rates as you move farther from the origin (not that I was stuck with that problem, or anything)
	InputVectors = [data[Input][0:localSteps], data[Input][0:localSteps]]
	TargetVector = (data[Target][i*localSteps:(i+1)*localSteps])
	print InputVectors[0]
	print TargetVector
	localWeights = 10* tools.genGaussWeights(InputVectors[0], 10)
	Weights = [numpy.average(TargetVector[0:localSteps/20]), 0.0]
	myLines.append(lines.Line2D([data[Input][i*localSteps], data[Input][(i+1)*localSteps]], [Weights[0], Weights[0] + Weights[1]*(InputVectors[0][-1])], color='g'))
	ax.add_line(myLines[-1])

	print " "
	print " "
	for j in range(1000):
		if (j%2 == 0):
			idx = j
		else:
			idx = 1000 - j
		# print "START"
		#Weights[0] = numpy.average(TargetVector)

		Weights = stochasticLWGradientDescentStep(Weights, localWeights, InputVectors, TargetVector, Alphas, j)
		err = tools.squareError(Weights, InputVectors, TargetVector)
		print "	Weights: " + str(Weights) 
		print "	Err: " + str(err)
		#print "using alphas " + str(Alphas)

		if (len(Weights) == 2):
			if (True):
				# Change the input vector offset to be where the data actually was
				InputVectorsReal = [data[Input][i*localSteps:(i+1)*localSteps], data[Input][i*localSteps:(i+1)*localSteps]]
				if j == 0:
					myLines.append(lines.Line2D([InputVectorsReal[0][0], InputVectorsReal[0][-1]], [Weights[0], Weights[0] + Weights[1]*(InputVectorsReal[0][-1])], color='y'))
					ax.add_line(myLines[-1])
					sums[i] += 1
				elif j == 999:
					myLines.append(lines.Line2D([InputVectorsReal[0][0], InputVectorsReal[0][-1]], [Weights[0], Weights[0] + Weights[1]*(InputVectorsReal[0][-1])], color='y'))
					ax.add_line(myLines[-1])
					sums[i] += 1
				else:
					myLines.append(lines.Line2D([InputVectorsReal[0][0], InputVectorsReal[0][-1]], [Weights[0], Weights[0] + Weights[1]*(InputVectorsReal[0][-1])], color=colors[i]))
					ax.add_line(myLines[-1])
					sums[i] += 1
			
		# else:
		# 	ys = tools.formSecondOrderPoly(xs, Weights)
		# 	plt.plot(xs, ys, color='r')

		if (record):	
			# if i > 2:
			# 	ax.lines[0].remove()
			# 	ax.lines[0].set_alpha(0.1)
			# 	ax.lines[1].set_alpha(0.5)

			name = './images/frame_'+str(f).zfill(pad)+'.png'
			f+=1
			plt.savefig(name, bbox_inches='tight', facecolor=(0.2,0.2,0.22))

print sums
print localSampleNum
plt.show()