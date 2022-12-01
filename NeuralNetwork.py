from Layer import Layer
from DataPoint import DataPoint


class NeuralNetwork:
	def __init__(self, layers):
		self.layers = [Layer(layers[x], layers[x+1]) for x in range(len(layers)-1)]

	def calculate_outputs(self, inputs):
		for layer in self.layers:
			inputs = layer.calculate_outputs(inputs)
		return inputs

	def classify(self, inputs):
		outputs = self.calculate_outputs(inputs)
		return outputs.index(max(outputs))

	def cost_of_point(self, inputPoint: DataPoint):
		outputs = self.calculate_outputs(inputPoint.inputs)
		cost = 0
		for i in range(len(outputs)):
			cost += self.layers[len(self.layers)-1].node_cost(outputs[i], inputPoint.evs[i])
		return cost

	def cost_of_set(self, inputPoints: list[DataPoint]):
		total_cost = 0
		for data_point in inputPoints:
			total_cost = self.cost_of_point(data_point)
		return total_cost / len(inputPoints)
