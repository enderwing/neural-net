import math as m
import random as r


def activation_function(weightedInput):
	return 1 / (1 + m.exp(-weightedInput))


def node_cost(outputActivation, expectedOutput):
	return (outputActivation - expectedOutput)**2


class Layer:
	def __init__(self, nodesIn: int, nodesOut: int):
		self.nodesIn = nodesIn
		self.nodesOut = nodesOut
		# (r.random() * 2 - 1) / m.sqrt(nodesIn)
		self.weights = [[(r.random() * 2 - 1) / m.sqrt(nodesIn) for i in range(nodesIn)] for j in range(nodesOut)]
		self.costGradientWeights = [[(r.random() * 2 - 1) / m.sqrt(nodesIn) for i in range(nodesIn)] for j in range(nodesOut)]
		self.biases = [0 for j in range(nodesOut)]
		self.costGradientBiases = [0 for j in range(nodesOut)]

	def calculate_outputs(self, inputs):
		outputs = [0] * self.nodesOut
		for iOut in range(self.nodesOut):
			sumOutput = self.biases[iOut]
			for iIn in range(self.nodesIn):
				sumOutput += inputs[iIn] * self.weights[iOut][iIn]
			outputs[iOut] = activation_function(sumOutput)
		return outputs

	def apply_gradients(self, learnRate):
		for out in range(self.nodesOut):
			for inn in range(self.nodesIn):
				self.weights[out][inn] -= self.costGradientWeights[out][inn] * learnRate
			self.biases[out] -= self.costGradientBiases[out] * learnRate
