import Layer
from DataPoint import DataPoint


class NeuralNetwork:
	def __init__(self, layers):
		self.layers = [Layer.Layer(layers[x], layers[x+1]) for x in range(len(layers)-1)]

	def calculate_outputs(self, inputs):
		for layer in self.layers:
			inputs = layer.calculate_outputs(inputs)
		return inputs

	def classify(self, point: DataPoint):
		outputs = self.calculate_outputs(point.inputs)
		return outputs.index(max(outputs))

	def cost_of_point(self, inputPoint: DataPoint):
		outputs = self.calculate_outputs(inputPoint.inputs)
		cost = 0
		for i in range(len(outputs)):
			cost += Layer.node_cost(outputs[i], inputPoint.evs[i])
		return cost

	def cost_average(self, inputPoints: list[DataPoint]):
		total_cost = 0
		for data_point in inputPoints:
			total_cost += self.cost_of_point(data_point)
		return total_cost / len(inputPoints)

	def learn(self, trainingData: list[DataPoint], learnRate):
		h = 0.0001
		originalCost = self.cost_average(trainingData)

		for layer in self.layers:
			for out in range(layer.nodesOut):
				for inn in range(layer.nodesIn):
					layer.weights[out][inn] += h
					costChange = self.cost_average(trainingData) - originalCost
					layer.weights[out][inn] -= h
					layer.costGradientWeights[out][inn] = costChange / h
			layer.biases[out] += h
			costChange = self.cost_average(trainingData) - originalCost
			layer.biases[out] -= h
			layer.costGradientBiases[out] = costChange / h
		for layer in self.layers:
			layer.apply_gradients(learnRate)

	def test_points(self, data: list[DataPoint]):
		count = 0
		for point in data:
			if self.classify(point) == point.evs[1]:
				count += 1
		return count
