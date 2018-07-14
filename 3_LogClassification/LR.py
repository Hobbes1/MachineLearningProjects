# Logistic Regression

import sys
import numpy as np
import random
import math 
from operator import attrgetter

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

sys.path.insert(0, '../common')
import tools

numClasses = 2 # number of classes (clusters)
numPoints = 200 # points per class
record = False
if sys.argv[1] == 'record':
	record = True

plotStyle = '2d'
if sys.argv[2] == '3':
	plotStyle = '3d'
##
# Form three (or CLI) gaussian PDFs with some points,
# these will be the three classifications to be made
##
i = 0
offsets = [(0, 0), (1.0, 1.0), (0, 0)]

classes = np.zeros(numPoints*numClasses)
xVals = np.zeros(numPoints*numClasses)
yVals = np.zeros(numPoints*numClasses)
for c in range(numClasses):
	for n in range (numPoints):
		classes[c*numPoints + n] = c
 		xVals[c*numPoints + n] = random.gauss(0, 0.3) + offsets[c][0]
 		yVals[c*numPoints + n] = random.gauss(0, 0.3) + offsets[c][1]

if plotStyle == '2d':
	plt.figure(0, figsize=(8, 7))
	plt.scatter(xVals, yVals, s=1)
	plt.axis([np.amin(xVals),
			  np.amax(xVals), 
			  np.amin(xVals), 
			  np.amax(yVals)])
	plt.gcf().set_facecolor((0.2, 0.2, 0.22))
	plt.gcf().figsize = (8, 6)
	ax = plt.subplot()
	ax.set_facecolor((0.2,0.2,0.22))
	ax.set_xlabel("X", color=(0.8,0.82,0.82), size=16)
	ax.set_ylabel("Y", color=(0.8,0.82,0.82), size=16)
	ax.spines['bottom'].set_color((0.8,0.82,0.82))
	ax.spines['left'].set_color((0.8,0.82,0.82))

	ax.spines['top'].set_color((0.2,0.2,0.22))
	ax.spines['right'].set_color((0.2,0.2,0.22))

	ax.tick_params(axis='x', colors=(0.8,0.82,0.82), size=3)
	ax.tick_params(axis='y', colors=(0.8,0.82,0.82), size=3)

#########################################################################################################
## Weights and algorithm loop
#########################################################################################################

#linear: theta0, theta1x, theta1y
#quad: theta0, theta1x, theta1y, theta2x
#circle/hyperbolic: theta0, theta1x, theta1y, theta2x, theta2y
alphas = [0.001, 0.001, 0.001, 0, 0]

weights = [
	0.0, #constant term
	-1.0, #x0 term 
	1.0, #x1 term
	0.0, #x0^2 term
	0.0, #x1^2 term
]

##
# where x is the inner product of weights and input vector
##
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

##
# one off inner prod for two parameter space
##
def innerProd2(weights, x0, x1):
	return weights[0] + \
		   weights[1] * x0 + \
   		   weights[2] * x1 + \
		   weights[3] * x0 * x0 + \
		   weights[4] * x1 * x1 

##
# for a particular weight theta, iterate through the "learning 
# algorithm" (gradient ascent). it's a one off for the two 
# parameter space case cuz i'm lazy
##
def gradientAscent2(x0s, x1s, classes, alphas, weights, index):
	#inner product term for the sigmoid
    res = weights[index]

    logLikelyhood = 0

    for i in range(len(classes)): 

	    # train the constant term:	
		if index == 0:
			logLikelyhood += alphas[0] * (classes[i] - sigmoid(innerProd2(weights, x0s[i], x1s[i])))
		# train the x0 (x) term:
		elif index == 1:
			logLikelyhood += alphas[1] * \
    						 (classes[i] - \
	                         sigmoid(innerProd2(weights, x0s[i], x1s[i]))) * x0s[i]

	    # train the x1 (y) term:
		elif index == 2:
			logLikelyhood += alphas[2] * \
    						 (classes[i] - \
	                         sigmoid(innerProd2(weights, x0s[i], x1s[i]))) * x1s[i]

    res += logLikelyhood
    print res
    return res

err = 0
steps = 90
pad = len(str(steps))
f=0
f3d = 0
colors = []
for i in range(numPoints*numClasses):
	colors.append('#bbe4e4')

size = 20

if (record):	
	# todo color points based on class
	if plotStyle == '2d':
		for i in range(numPoints*numClasses):
			if sigmoid(innerProd2(weights, xVals[i], yVals[i])) > 0.5:
				colors[i] = '#bbe4e4'
			else:
				colors[i] = '#cc383c'

		plt.figure(0, figsize=(8, 7))
		plt.scatter(xVals, yVals, c=colors, s=1)
		name = './images/frame_'+str(f).zfill(pad)+'.png'
		f+=1
		plt.savefig(name, bbox_inches='tight', facecolor=(0.2,0.2,0.22))

	elif plotStyle == '3d':
		xs = np.arange(np.amin(xVals), np.amax(xVals), (np.amax(xVals)-np.amin(xVals))/100)
		ys = np.zeros(100)
		myLines = []

		startx = -1.0
		endx = 3.0
		starty = -1.0
		endy = 3.0
		xs = np.zeros([size,size])
		ys = np.zeros([size,size])
		zs = np.zeros([size, size])
		val = 0.0

		for xIdx in range((size)):
			for yIdx in range((size)):
				xs[xIdx][yIdx] = (endx - startx) / size * xIdx + startx
				ys[xIdx][yIdx] = (endy - starty) / size * yIdx + starty
		for xIdx in range((size)):
			for yIdx in range((size)):
				val = weights[0]+xs[xIdx][yIdx]*weights[1]+ys[xIdx][yIdx]*weights[2]
				zs[xIdx][yIdx] = sigmoid(val)

		fig3d = plt.figure(2)
		ax3d = fig3d.add_subplot(111, projection='3d')
		ax3d.view_init(elev=60, azim=270)
		ax3d.plot_surface(xs, ys, zs, cmap=cm.coolwarm)

for i in range(steps):
	for j in range(len(weights)):
		weights[j] = gradientAscent2(xVals, yVals, classes, alphas, weights, j)

	print ""

	# color and save 
	if (record):	
		# todo color points based on class
		if plotStyle == '2d':
			for i in range(numPoints*numClasses):
				if sigmoid(innerProd2(weights, xVals[i], yVals[i])) > 0.5:
					colors[i] = '#bbe4e4'
				else:
					colors[i] = '#cc383c'

			plt.figure(0, figsize=(8, 7))
			plt.scatter(xVals, yVals, c=colors, s=1)
			name = './images/frame_'+str(f).zfill(pad)+'.png'
			f+=1
			plt.savefig(name, bbox_inches='tight', facecolor=(0.2,0.2,0.22))

		elif plotStyle == '3d':
			for xIdx in range((size)):
				for yIdx in range((size)):
					val = weights[0]+xs[xIdx][yIdx]*weights[1]+ys[xIdx][yIdx]*weights[2]
					zs[xIdx][yIdx] = sigmoid(val)

			print zs
			print "wow"
			ax3d = fig3d.add_subplot(111, projection='3d')
			ax3d.plot_surface(xs, ys, zs, cmap=cm.coolwarm)
			name = './images/frame3d_'+str(f3d).zfill(pad)+'.png'
			ax3d.view_init(elev = 60, azim = 270)
			f3d+=1
			plt.savefig(name, bbox_inches='tight', facecolor=(0.2,0.2,0.22))
			ax3d.clear()

plt.show()