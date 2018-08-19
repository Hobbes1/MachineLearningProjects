import numpy as np

class GDAData(object):
	'''
	Class container for all data associated with GDA classification.
	It just creates gaussian distributions.
	'''
	distA = np.zeros(1)
	distB = np.zeros(1)

	def __init__(self):
		self.distA = 0
		self.distB = 0

	def load(self, means, covariances, nums):
		'''
		Load (two) gaussian distributions into arrays 
		mDataA and mDataB.
		@param means 		2 lists of 2 variables corresponding to 2 x,y means for 2 distributions
		@param covariances  2, 2x2 arrays corresponding to 2 covariance arrays for 2 distributions
		@param nums 		the number of points to generate for two distributions
		'''

		#Let numpy do bounds checking for us
		self.distA = np.random.multivariate_normal(means[0], covariances[0], nums[0])
		self.distB = np.random.multivariate_normal(means[1], covariances[1], nums[1])

		#Squish them a little to some polynomial, as to not be approximating gaussians with gaussians
		for x in np.nditer(self.distA[:,0], op_flags=['readwrite']):
			x[...] = 0.95*x - 0.003*x*x*x

		#Probably doable with iterators
		i = 0
		for y in np.nditer(self.distA[:,1], op_flags=['readwrite']):
			y[...] = 1.05*y - 0.02*y*y
			i += 1
			if (y < 4):
				self.distA[i-1,0] += y/4

		for x in np.nditer(self.distB[:,0], op_flags=['readwrite']):
			x[...] = 0.95*x - 0.2*x*x + 10

		for y in np.nditer(self.distB[:,1], op_flags=['readwrite']):
			y[...] = 1.05*y - 0.2/(y*y)