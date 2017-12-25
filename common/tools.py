
import sys
import numpy
from numpy.lib import recfunctions
import datetime
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import lines

####################################
# A collection of tools which are used generally
# within these regressions. A lot of them involve / will involve
# data scrubbing. 
# The actual regression / "learning" algorithms will be kept within 
# their respective project files.

		# Tuning alpha parameters is easier if target vectors
		# are normalized

def normalizeVector(vector):
	summ = numpy.sum(vector)
	for i in range(len(vector)):
		vector[i] = vector[i]/summ

	return vector

		# Some of my dataset had negative values which are invalid
		# I chose to interpolate using neighboring values, rather 
		# than deleting and reshaping. For many negaives in a row
		# this just means putting reassigning them to be on a straight line
		# between the two closest positive points.

def interpolateBadNegatives(dataVector):
	for idx in range(len(dataVector)):
		if dataVector[idx] < 0:
			print ("found a negative")
			for negIdx in range(len(dataVector) - idx):
				if dataVector[idx + negIdx] < 0:
					print ("found a negative after the negative: " + str(negIdx))
					continue
				else:
					print("Reassigning: " + str(negIdx))
					for negativeIndex in range(negIdx):
						dataVector[idx + negativeIndex] = dataVector[idx - 1] + (negativeIndex * (dataVector[negIdx] - dataVector[idx-1])/(negIdx-idx))
						print("reassigned negative to: " + str(dataVector[idx]))
					break

	return dataVector

		# Load a numpy dataset from file with dataType sets and 
		# columns sets specified for use. Print it to make sure im not being dumb.

def loadAndPrintData(filename, dtypes, usecolumns):
	data = numpy.genfromtxt(filename, delimiter=',', names=True, dtype=dtypes, usecols=usecolumns)
	print data.shape
	print "	Data set:"
	print "%-20s %-20s %-20s %-40s" % ("\tColumn Name", "Number of Elements", "Data Type", "Nulls found")
	for name in data.dtype.names:
		print "%-20s %-20s %-20s %-40s" % ("\t"+name, str(len(data[name])), data[name].dtype, "and TODO number of nulls")

	print data.shape
	return data

		# Given a dataset with year-month-day-hour columns
		# Merge those into a single form (total days)
		# and reshape the dataset. TODO should probably 
		# make this more generic to take all timestamp forms in the future
		# It also adds a column for time, in hours, from the initial measurement

def mergeTimeColumns(data, years, months, days, hours):
	datetimes = []
	hours2 = []
	for idx in range(data.shape[0]):
		datetimes.append(datetime.datetime(years[idx], months[idx], days[idx], hours[idx]))
		hours2.append((datetimes[idx] - datetimes[0]).total_seconds() / 3600.0)

	data = recfunctions.append_fields(data, data=datetimes, names="datetimes", dtypes='M8[us]')
	data = recfunctions.append_fields(data, data=hours2, names="hours", dtypes='int64')

	data = recfunctions.drop_fields(data, drop_names=('year', 'month', 'day', 'hour'))
	print "	Data set after merging time columns:"

	print "%-20s %-20s %-20s %-40s" % ("\tColumn Name", "Number of Elements", "Data Type", "Nulls found")
	for name in data.dtype.names:
		print "%-20s %-20s %-20s %-40s" % ("\t"+name, str(len(data[name])), data[name].dtype, "and TODO number of nulls")

	return data

		# This is not used explicitely in the regression alg.
		# but is still useful to view error as a function 
		# of regression steps taken

def formSecondOrderPoly(xs, Weights):
	ys = numpy.zeros(100)
	for i in range(len(xs)):
		ys[i] = Weights[0] + xs[i] * Weights[1] + xs[i] * xs[i] * Weights[2]

	return ys

def squareError(Weights, InputVectors, TargetVector):

	err = 0.0
	for weightSet in range(len(Weights)):
		tempGuess = 0.0
		for idx in range(len(InputVectors[0])):
			tempGuess = Weights[0] * InputVectors[0][idx] #h_theta(X) from Andrew's lecture
			err += (tempGuess - TargetVector[idx]) ** 2

	return err

		# Performing sum of derivs for each order of polynomial asked for
		# Higher order polys get more derivatives and terms added to all weightDerivatives

def firstOrderPolyErrors(weightIdx, vectorIdx, Weights, InputVectors, TargetVector):
	sumOfDerivativeErrors = 0
	if weightIdx == 0:
		sumOfDerivativeErrors += ((Weights[0] + Weights[1] * InputVectors[weightIdx][vectorIdx] - TargetVector[vectorIdx]))
	elif weightIdx == 1:
		sumOfDerivativeErrors += (Weights[0] + Weights[1] * InputVectors[weightIdx][vectorIdx] - TargetVector[vectorIdx]) * InputVectors[weightIdx][vectorIdx]

	return sumOfDerivativeErrors

def secondOrderPolyErrors(weightIdx, vectorIdx, Weights, InputVectors, TargetVector):
	sumOfDerivativeErrors = 0
	if weightIdx == 0:
		sumOfDerivativeErrors += ((Weights[0] + Weights[1] * InputVectors[0][vectorIdx] + Weights[2] * InputVectors[0][vectorIdx] * InputVectors[0][vectorIdx] - TargetVector[vectorIdx]))
	elif weightIdx == 1:
		sumOfDerivativeErrors += (Weights[0] + Weights[1] * InputVectors[0][vectorIdx] + Weights[2] * InputVectors[0][vectorIdx] * InputVectors[0][vectorIdx] - TargetVector[vectorIdx]) * InputVectors[0][vectorIdx]
	elif weightIdx == 2:
		sumOfDerivativeErrors += (Weights[0] + Weights[1] * InputVectors[0][vectorIdx] + Weights[2] * InputVectors[0][vectorIdx] * InputVectors[0][vectorIdx] - TargetVector[vectorIdx]) * (InputVectors[0][vectorIdx] * InputVectors[0][vectorIdx])

	return sumOfDerivativeErrors

		# TODO not implemented yet (copy pasted from 2nd order)

def thirdOrderPolyErrors(Weights, InputVectors, TargetVector):
	sumOfDerivativeErrors = 0
	if weightIdx == 0:
		sumOfDerivativeErrors += ((Weights[0] + Weights[1] * InputVectors[weightIdx][idx] + Weights[2] * InputVectors[weightIdx][vectorIdx] * InputVectors[weightIdx][vectorIdx] - TargetVector[idx]))
	elif weightIdx == 1:
		sumOfDerivativeErrors += (Weights[0] + Weights[1] * InputVectors[weightIdx][idx] + Weights[2] * InputVectors[weightIdx][vectorIdx] * InputVectors[weightIdx][vectorIdx] - TargetVector[idx]) * InputVectors[weightIdx][idx]
	elif weightIdx == 2:
		sumOfDerivativeErrors += (Weights[0] + Weights[1] * InputVectors[weightIdx][idx] + Weights[2] * InputVectors[weightIdx][vectorIdx] * InputVectors[weightIdx][vectorIdx] - TargetVector[idx]) * InputVectors[weightIdx][idx] * InputVectors[weightIdx][idx]

	return sumOfDerivativeErrors