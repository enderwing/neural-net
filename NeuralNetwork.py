import os

import Layer
from DataPoint import DataPoint


class NeuralNetwork:
	# init network, pick an ID that is not present in the list of saved networks
	def __init__(self, layers):
		self.layers = [Layer.Layer(layers[x], layers[x+1], True if x == len(layers)-2 else False) for x in range(len(layers)-1)]
		networkNames = os.listdir("networks/")
		testID = 0
		for i in range(len(networkNames)):
			if f"network{testID}.pickle" in networkNames:
				testID += 1
		self.networkID = testID

	# returns the output node values for a set of inputs
	def calculate_outputs(self, inputs):
		for layer in self.layers:
			inputs = layer.calculate_outputs(inputs)
		return inputs

	# returns the specific output node that had the highest output value for a set of inputs
	def classify(self, point: DataPoint):
		outputs = self.calculate_outputs(point.inputs)
		return outputs.index(max(outputs))

	# check if a DataPoint is classified correctly (relies on the DataPoint having correctly set evs)
	# NOTE: set up to work for specific data used to diagnose the network, not for general use
	def is_classified_correctly(self, point: DataPoint):
		if self.classify(point) == point.evs[1]:
			return True
		return False

	# return the cost of a point (the cost of each output node of the point summed together)
	def cost_of_point(self, inputPoint: DataPoint):
		outputs = self.calculate_outputs(inputPoint.inputs)
		cost = 0
		for i in range(len(outputs)):
			cost += Layer.node_cost(outputs[i], inputPoint.evs[i])
		return cost

	# return info about a point (written expecting the data to be an image, but could be generalized)
	def point_info(self, point: DataPoint):
		outputs = self.calculate_outputs(point.inputs)
		cost = 0
		for i in range(len(outputs)):
			cost += Layer.node_cost(outputs[i], point.evs[i])
		return outputs.index(max(outputs)) == point.evs[10], cost

	# runs the above on a list and returns aggregate data
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

	# average cost of a list of DataPoints
	def cost_average(self, inputPoints: list[DataPoint]):
		total_cost = 0
		for data_point in inputPoints:
			total_cost += self.cost_of_point(data_point)
		return total_cost / len(inputPoints)

	# the main learn function that uses slow, approximations to learn
	# DEPRECATED
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

	# reset the weight and bias gradients of all layers to 0
	def reset_gradients(self):
		for layer in self.layers:
			layer.costGradientWeights = [[0] * layer.nodesIn] * layer.nodesOut
			layer.costGradientBiases = [0] * layer.nodesOut

	# main learn function, takes a list point and does backpropagation after running each point to train the network
	def learn_with_derivatives(self, trainingData: list[DataPoint], learnRate):
		for point in trainingData:
			self.derivative_gradient_update(point)
			for layer in self.layers:
				layer.apply_gradients(learnRate)
		# self.reset_gradients()

	# run a single point and increment the gradient trackers for each weight and bias in the network
	def derivative_gradient_update(self, point: DataPoint):
		classification = self.is_classified_correctly(point)

		# do the output layer first
		outputLayer = self.layers[len(self.layers)-1]
		nodeValues = outputLayer.calculate_output_node_values(point.evs)
		outputLayer.update_gradients(nodeValues, classification)

		# iterate through hidden layers, passing node values along
		for i in range(len(self.layers) - 1):
			hiddenLayerIndex = len(self.layers) - 2 - i
			hiddenLayer = self.layers[hiddenLayerIndex]
			nodeValues = hiddenLayer.calculate_hidden_node_values(self.layers[hiddenLayerIndex + 1], nodeValues)
			hiddenLayer.update_gradients(nodeValues, classification)

	# test a list of 2-input points
	# OLD, used for diagnosing the network
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

	# check a single DataPoint containing info about a 28x28 image of a digit
	def check_digit(self, digit: DataPoint):
		if self.classify(digit) == digit.evs[10]:
			return True
		return False

	# run the above on a list of DataPoints and return agregate data
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
