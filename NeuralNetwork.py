import os

import Layer
from DataPoint import DataPoint


class NeuralNetwork:
	def __init__(self, layers):
		self.layers = [Layer.Layer(layers[x], layers[x+1], True if x == len(layers)-2 else False) for x in range(len(layers)-1)]
		networkNames = os.listdir("networks/")
		testID = 0
		for i in range(len(networkNames)):
			if f"network{testID}.pickle" in networkNames:
				testID += 1
		self.networkID = testID

	def calculate_outputs(self, inputs):
		for layer in self.layers:
			inputs = layer.calculate_outputs(inputs)
		return inputs

	def classify(self, point: DataPoint):
		outputs = self.calculate_outputs(point.inputs)
		return outputs.index(max(outputs))

	def is_classified_correctly(self, point: DataPoint):
		if self.classify(point) == point.evs[1]:
			return True
		return False

	def cost_of_point(self, inputPoint: DataPoint):
		outputs = self.calculate_outputs(inputPoint.inputs)
		cost = 0
		for i in range(len(outputs)):
			cost += Layer.node_cost(outputs[i], inputPoint.evs[i])
		return cost

	def point_info(self, point: DataPoint):
		outputs = self.calculate_outputs(point.inputs)
		cost = 0
		for i in range(len(outputs)):
			cost += Layer.node_cost(outputs[i], point.evs[i])
		return outputs.index(max(outputs)) == point.evs[10], cost

	def points_info(self, points: list[DataPoint]):
		costTotal = 0
		countCorrect = 0
		failed = []
		passed = []
		for point in points:
			correct, cost = self.point_info(point)
			costTotal += cost

			if correct:
				countCorrect += 1
				passed.append(point)
			else:
				failed.append(point)
		return costTotal/len(points), countCorrect, passed, failed

	def cost_average(self, inputPoints: list[DataPoint]):
		total_cost = 0
		for data_point in inputPoints:
			total_cost += self.cost_of_point(data_point)
		return total_cost / len(inputPoints)

	def learn(self, trainingData: list[DataPoint], learnRate):
		h = 0.0000000001
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

	def reset_gradients(self):
		for layer in self.layers:
			layer.costGradientWeights = [[0] * layer.nodesIn] * layer.nodesOut
			layer.costGradientBiases = [0] * layer.nodesOut

	def learn_with_derivatives(self, trainingData: list[DataPoint], learnRate):
		for point in trainingData:
			self.derivative_gradient_update(point)
			for layer in self.layers:
				layer.apply_gradients(learnRate)
		# self.reset_gradients()

	def derivative_gradient_update(self, point: DataPoint):
		classification = self.is_classified_correctly(point)

		outputLayer = self.layers[len(self.layers)-1]
		nodeValues = outputLayer.calculate_output_node_values(point.evs)
		outputLayer.update_gradients(nodeValues, classification)

		for i in range(len(self.layers) - 1):
			hiddenLayerIndex = len(self.layers) - 2 - i
			hiddenLayer = self.layers[hiddenLayerIndex]
			nodeValues = hiddenLayer.calculate_hidden_node_values(self.layers[hiddenLayerIndex + 1], nodeValues)
			hiddenLayer.update_gradients(nodeValues, classification)

	def test_points(self, data: list[DataPoint]):
		count = 0
		failedPoints = []
		passedPoints = []
		for point in data:
			if self.classify(point) == point.evs[1]:
				count += 1
				passedPoints.append(point)
			else:
				failedPoints.append(point)
		return count, passedPoints, failedPoints

	def check_digit(self, digit: DataPoint):
		if self.classify(digit) == digit.evs[10]:
			return True
		return False

	def check_digits(self, digits: list[DataPoint]):
		count = 0
		failed = []
		passed = []
		for i in range(len(digits)):
			if self.check_digit(digits[i]):
				count += 1
				passed.append(digits[i])
			else:
				failed.append(digits[i])
		return count, passed, failed

