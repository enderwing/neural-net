import math

def activation_function(weightedInput):
	return 1 / (1 + math.exp(-weightedInput))


class Layer:
	def __init__(self, nodesIn: int, nodesOut: int):
		self.nodesIn = nodesIn
		self.nodesOut = nodesOut
		self.weights = [[0] * nodesIn] * nodesOut
		self.biases = [0] * nodesOut

	def node_cost(self, outputActivation, expectedOutput):
		return (outputActivation - expectedOutput)**2

	def calculate_outputs(self, inputs):
		outputs = [0] * self.nodesOut
		for iOut in range(self.nodesOut):
			sumOutput = self.biases[iOut]
			for iIn in range(self.nodesIn):
				sumOutput += inputs[iIn] * self.weights[iOut][iIn]
			outputs[iOut] = activation_function(sumOutput)
		return outputs
