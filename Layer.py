import math as m
import random as r
import numpy as np


## below are activation functions, which transform weighted inputs into the outputs of the node
# and the node cost function, which defines how far off a node is from the expected value of a set of inputs

def activation_function_sigmoid(weightedInput):
	## sigmoid
	if weightedInput < -700:
		return 0
	return 1 / (1 + np.exp(-weightedInput))

def activation_function_ReLU(weightedInput):
	# ReLU
	return max(0, weightedInput)

def activation_derivative_sigmoid(weightedInput):
	## sigmoid
	activation = activation_function_sigmoid(weightedInput)
	return activation * (1 - activation)

def activation_derivative_ReLU(weightedInput):
	## ReLU
	if weightedInput <= 0:
		return 0
	return 1

def node_cost(outputActivation, expectedOutput):
	return (outputActivation - expectedOutput)**2

def node_cost_derivative(activation, expectedOutput):
	return 2 * (activation - expectedOutput)


class Layer:
	# setup all the variables needed to store data about the node, weights and biases, and intermediary values
	def __init__(self, nodesIn: int, nodesOut: int, isOut: bool):
		self.nodesIn = nodesIn
		self.nodesOut = nodesOut
		self.isOutputLayer = isOut
		# (r.random() * 2 - 1) / m.sqrt(nodesIn)
		self.weights = [[(r.random() * 2 - 1) / m.sqrt(nodesIn) for i in range(nodesIn)] for j in range(nodesOut)]
		self.costGradientWeights = [[(r.random() * 2 - 1) / m.sqrt(nodesIn) for i in range(nodesIn)] for j in range(nodesOut)]
		self.biases = [0 for j in range(nodesOut)]
		self.costGradientBiases = [0 for j in range(nodesOut)]
		self.activationValues = [0] * nodesOut
		self.weightedInputs = [0] * nodesOut
		self.previousLayerActivations = [0] * nodesIn

	# calculate weighted inputs from inputs, pass through the activation function, return outputs
	def calculate_outputs(self, inputs):
		self.previousLayerActivations = inputs
		outputs = [0] * self.nodesOut
		for iOut in range(self.nodesOut):
			weightedInput = self.biases[iOut]
			for iIn in range(self.nodesIn):
				weightedInput += inputs[iIn] * self.weights[iOut][iIn]
			self.weightedInputs[iOut] = weightedInput
			if self.isOutputLayer:
				self.activationValues[iOut] = activation_function_sigmoid(weightedInput)
			else:
				self.activationValues[iOut] = activation_function_ReLU(weightedInput)
			outputs[iOut] = self.activationValues[iOut]
		return outputs

	# part of backpropagation
	# calculate how sensitive the cost is to a change in the weighted input of the output nodes
	def calculate_output_node_values(self, evs):
		nodeValues = [0] * self.nodesOut
		for i in range(self.nodesOut):
			nodeValues[i] = activation_derivative_sigmoid(self.weightedInputs[i]) * node_cost_derivative(self.activationValues[i], evs[i])
		return nodeValues

	# part of backpropagation
	# calculate how sensitive the cost is to a change in the weighted input of the hidden nodes
	def calculate_hidden_node_values(self, prevLayer, prevLayerNodeValues):
		hiddenNodeValues = [0] * self.nodesOut
		for iOut in range(self.nodesOut):
			nodeValue = 0
			for iPrev in range(prevLayer.nodesOut):
				nodeValue += prevLayer.weights[iPrev][iOut] * prevLayerNodeValues[iPrev]
			hiddenNodeValues[iOut] = nodeValue * activation_derivative_ReLU(self.weightedInputs[iOut])
		return hiddenNodeValues

	# part of backpropagation
	# using the node values of the current layer, calculate and store gradients of this layers weights and biases
	def update_gradients(self, nodeValues, correct: bool):
		for iOut in range(self.nodesOut):
			for iIn in range(self.nodesIn):
				costWrtWeight = self.previousLayerActivations[iIn] * nodeValues[iOut]
				self.costGradientWeights[iOut][iIn] = costWrtWeight * (2 if not correct else 1)
			self.costGradientBiases[iOut] = nodeValues[iOut] * (2 if not correct else 1)

	# part of backpropagation
	# apply accumulated gradients, scaled by the learn rate
	def apply_gradients(self, learnRate):
		for iOut in range(self.nodesOut):
			for iIn in range(self.nodesIn):
				self.weights[iOut][iIn] -= self.costGradientWeights[iOut][iIn] * learnRate
			self.biases[iOut] -= self.costGradientBiases[iOut] * learnRate
