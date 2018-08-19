import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

from GDA_Data import GDAData
from GDA_Classifier import GDAClassifier
import sys

means = [
	[5,5], 
	[7.5,8.7]
	]

covariances = [
	[[2.5,0],[0,2]],
	[[1, -0.6],[-0.6, 1]]
	]

numA = 2000
numB = 1000
nums = [numA, numB]

doPlots = False
if sys.argv[1] == 'plot':
	doPlots = True

#ensure same distribution each time
np.random.seed(999)

data = GDAData()
data.load(means, covariances, nums)

testNumA = 6
testNumB = 3

howMany = 60
resultVector = np.zeros((howMany, 5), dtype=np.object)

for i in range(howMany):
	classifier = GDAClassifier(testNumA, testNumB, data.distA, data.distB)
	results = classifier.classifyPoints()
	print(str.format('{0:.4f}',((results.numCorrectA - results.numIncorrectA) / (results.numCorrectA + results.numIncorrectA))*100.0) + "% on classA and " + \
		  str.format('{0:.4f}',((results.numCorrectB - results.numIncorrectB) / (results.numCorrectB + results.numIncorrectB))*100.0) + "% on classB") 

	percentA = (results.numCorrectA - results.numIncorrectA) / (results.numCorrectA + results.numIncorrectA)*100.0
	percentB = (results.numCorrectB - results.numIncorrectB) / (results.numCorrectB + results.numIncorrectB)*100.0
	testNumA += 2
	testNumB += 1
	resultVector[i,:] = [testNumA+testNumB, percentA, percentB, results.gaussDistA, results.gaussDistB]


if (doPlots):

	fig1 = plt.figure(0, figsize=(8, 7))
	plt.axis([0, 12, 0, 12])
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
	ax.scatter(data.distA[:,0], data.distA[:,1], c='c', s=0.5)
	ax.scatter(data.distB[:,0], data.distB[:,1], c='r', s=0.5)
	ax.set_axis_off()
	fig1.savefig("testDist.png", transparent=True)

	################################################################
	pad = len(str(howMany))
	for f in range(howMany):

		numSamples = 256
		xs = np.linspace(0.0, 12.0, numSamples)
		ys = np.linspace(0.0, 12.0, numSamples)

		gaussDistA = resultVector[f,4]
		gaussDistB = resultVector[f,3]

		gaussFieldA = np.zeros((numSamples, numSamples))
		gaussFieldB = np.zeros((numSamples, numSamples))
		for i in range(len(xs)):
			for j in range(len(ys)):
				gaussFieldA[i,numSamples - 1 - j] = gaussDistA.pdf((xs[i], ys[j]))
				gaussFieldB[i,numSamples - 1 - j] = gaussDistB.pdf((xs[i], ys[j]))

		fig2 = plt.figure(1, figsize=(8, 7), frameon=False)
		ax = plt.Axes(fig2, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig2.add_axes(ax)
		fig2.set_facecolor((0.2, 0.2, 0.22))
		ax.imshow(gaussFieldA.T + gaussFieldB.T, cmap='plasma')
		#plt.imshow(gaussFieldA.T + gaussFieldB.T, cmap='plasma')
		name = 'images/frame_'+str(f).zfill(pad)+'.png'
		print(name)
		fig2.savefig(name, transparent=True, bbox_inches='tight')

	###################################################################

	fig3= plt.figure(2, figsize=(8, 7))
	#fig3.patch.set_facecolor('b')
	fig3.patch.set_alpha(0.0)
	ax3 = plt.subplot()
	plt.axis([9, resultVector[-1,0], 0, 100.5])
	plt.gcf().set_facecolor((0.2, 0.2, 0.22))
	plt.gcf().figsize = (8, 6)
	ax = plt.subplot()
	#ax.set_facecolor(transparent=True)
	ax.set_xlabel("Number of points included in model", color=(0.2,0.22,0.22), size=16)
	ax.set_ylabel("Percent accuracy of following predictions", color=(0.2,0.22,0.22), size=16)
	ax.spines['bottom'].set_color((0.2,0.22,0.22))
	ax.spines['left'].set_color((0.2,0.22,0.22))
	ax.spines['top'].set_color((0.2,0.22,0.22))
	ax.spines['right'].set_color((0.2,0.22,0.22))
	ax.tick_params(axis='x', colors=(0.2,0.22,0.22), size=3)
	ax.tick_params(axis='y', colors=(0.2,0.22,0.22), size=3)
	ax.plot(resultVector[:,0], resultVector[:,1], c='c', label='Class A')
	ax.plot(resultVector[:,0], resultVector[:,2], c='r', label='Class B')
	ax.plot(resultVector[:,0], (2*resultVector[:,1]+resultVector[:,2])/3.0, c='m', label='Total')
	leg = plt.legend(facecolor=(1.0,1.0,1.0,0.0), edgecolor=(0.2,0.22,0.22,0.0))
	for text in leg.get_texts():
		text.set_color((0.2,0.22,0.22))

	plt.tight_layout()

	fig3.savefig("test.png", transparent=True)
	plt.show()