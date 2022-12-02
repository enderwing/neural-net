import math as m
import random as r


def activation_function(weightedInput):
	return 1 / (1 + m.exp(-weightedInput))

def activation_derivative(weightedInput):
	activation = activation_function(weightedInput)
	return activation * (1 - activation)

def node_cost(outputActivation, expectedOutput):
	return (outputActivation - expectedOutput)**2

def node_cost_derivative(activation, expectedOutput):
	return 2 * (activation - expectedOutput)


class Layer:
	def __init__(self, nodesIn: int, nodesOut: int):
		self.nodesIn = nodesIn
		self.nodesOut = nodesOut
		# (r.random() * 2 - 1) / m.sqrt(nodesIn)
		self.weights = [[(r.random() * 2 - 1) / m.sqrt(nodesIn) for i in range(nodesIn)] for j in range(nodesOut)]
		self.costGradientWeights = [[(r.random() * 2 - 1) / m.sqrt(nodesIn) for i in range(nodesIn)] for j in range(nodesOut)]
		self.biases = [0 for j in range(nodesOut)]
		self.costGradientBiases = [0 for j in range(nodesOut)]
		self.activationValues = [0] * nodesOut
		self.weightedInputs = [0] * nodesOut
		self.inputs = [0] * nodesIn

	def calculate_outputs(self, inputs):
		self.inputs = inputs
		outputs = [0] * self.nodesOut
		for iOut in range(self.nodesOut):
			weightedInput = self.biases[iOut]
			for iIn in range(self.nodesIn):
				weightedInput += inputs[iIn] * self.weights[iOut][iIn]
			self.weightedInputs[iOut] = weightedInput
			self.activationValues[iOut] = activation_function(weightedInput)
			outputs[iOut] = self.activationValues[iOut]
		return outputs

	def calculate_output_node_values(self, evs):
		nodeValues = [0] * self.nodesOut
		for i in range(self.nodesOut):
			nodeValues[i] = activation_derivative(self.weightedInputs[i]) * node_cost_derivative(self.activationValues[i], evs[i])
		return nodeValues

	def calculate_hidden_node_values(self, prevLayer, prevLayerNodeValues):
		hiddenNodeValues = [0] * self.nodesOut
		for iOut in range(self.nodesOut):
			nodeValue = 0
			for iPrev in range(prevLayer.nodesOut):
				nodeValue += prevLayer.weights[iPrev][iOut] * prevLayerNodeValues[iPrev]
			hiddenNodeValues[iOut] = nodeValue * activation_derivative(self.weightedInputs[iOut])
		return hiddenNodeValues

	def update_gradients(self, nodeValues):
		for iOut in range(self.nodesOut):
			for iIn in range(self.nodesIn):
				costWrtWeight = self.inputs[iIn] * nodeValues[iOut]
				self.costGradientWeights[iOut][iIn] += costWrtWeight
			self.costGradientBiases[iOut] += nodeValues[iOut]

	def apply_gradients(self, learnRate):
		for out in range(self.nodesOut):
			for inn in range(self.nodesIn):
				self.weights[out][inn] -= self.costGradientWeights[out][inn] * learnRate
			self.biases[out] -= self.costGradientBiases[out] * learnRate
