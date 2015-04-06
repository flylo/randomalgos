import numpy as np
import pdb
from numpy import shape, ones, concatenate, transpose, dot
from numpy.linalg import inv

class linreg(object):
	"""
	implementation of Linear Regression from:
		Machine Learning:  An Algorithmic Perspective
	link to original code: https://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html
	beta coefficients calculated as follows, where X is input matrix
		beta = ((X**T * X)**-1) * X**T * t
	"""
	def __init__(self, inputs, targets):
		self.inputs = inputs
		self.targets = targets

	def inputMatrixTransform(self):
		print "Transforming Matrix for suitable analysis"
		input_rows = shape(self.inputs)[0]
		ones_vec = -ones((input_rows, 1))
		self.X = concatenate((self.inputs, ones_vec), axis=1)

	def calculateBeta(self):
		print "Calculating the Beta Coefficients"
		Xtranspose_X = dot(transpose(self.X), self.X)
		Xtranspose_X_minusONE = inv(Xtranspose_X)
		Xtranspose_X_minusONE_Xtranspose = dot(Xtranspose_X_minusONE,
			transpose(self.X))
		self.beta = dot(Xtranspose_X_minusONE_Xtranspose, self.targets)

	def predictOutputs(self):
		print "Generating Predicted Outputs"
		self.outputs = dot(self.X, self.beta)

	def run(self):
		self.inputMatrixTransform()
		self.calculateBeta()
		self.predictOutputs()


if __name__ == '__main__':
	inputs = np.random.rand(10e7, 5)
	targets = np.random.rand(10e7)
	reg = linreg(inputs, targets)
	reg.run()
	pdb.set_trace()
