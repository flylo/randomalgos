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
		input_rows = shape(self.inputs)[0]
		ones_vec = -ones((input_rows, 1))
		self.X = concatenate((self.inputs, ones_vec), axis=1)

	def calculateBeta(self):
		#self.XT = transpose(self.X)
		Xtranspose_X = dot(transpose(self.X), self.X)
		Xtranspose_X_minusONE = inv(Xtranspose_X)
		Xtranspose_X_minusONE_Xtranspose = dot(Xtranspose_X_minusONE,
			transpose(self.X))
		self.beta = dot(Xtranspose_X_minusONE_Xtranspose, self.targets)

	def predictOutputs(self):
		self.outputs = dot(self.X, self.beta)

	def run(self):
		self.inputMatrixTransform()
		self.calculateBeta()
		self.predictOutputs()


if __name__ == '__main__':
	#inputs = np.array([[1], [2], [1], [3]])
	inputs = np.array([[1, 3, 2], [2, 4, 4], [1, 1, 2], [3, 3, 4]])
	targets = np.array([[2], [3], [3], [4]])
	reg = linreg(inputs, targets)
	reg.run()
	pdb.set_trace()
