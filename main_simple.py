import copy
import math as m
import pickle
import random as r
import time as t
import matplotlib.pyplot as plt
import numpy as np

from DataPoint import DataPoint
from NeuralNetwork import NeuralNetwork
import NNetWrapper as nwrap


# old code used for testing the net with simple, 2 parameter data
# could visualize data by graphing in a coordinate grid with color coordination

xrange = 1
yrange = 1
radius = 0.3

def generate_data(amount):
	data = []
	for i in range(amount):
		point = [r.random() * xrange, r.random() * yrange]
		# check = 1 if m.sqrt(point[0]**2 + point[1]**2) < radius else 0
		check = 1 if point[1] < radius else 0
		data.append(DataPoint(point, [check, 0 if check else 1]))
	return data

def generate_data_in_grid():
	xmin = -10
	xmax = 10
	xinc = 1
	ymin = -10
	ymax = 10
	yinc = 1

	xval = xmin
	yval = ymin
	data = []
	while xval <= xmax + 0.01:
		while yval <= ymax + 0.01:
			point = [xval,  yval]
			# check = 1 if m.sqrt(point[0] ** 2 + point[1] ** 2) < radius else 0
			check = 1 if point[1] < radius else 0
			data.append(DataPoint(point, [check, 0 if check else 1]))
			yval += yinc
		xval += xinc
		yval = ymin
	return data

def graph_data(data: list[DataPoint], nnet: NeuralNetwork):
	theta = np.arange(0, np.pi/2, 0.01)
	# plt.plot(radius * np.cos(theta), radius * np.sin(theta))
	for i in range(len(data)):
		outcome = nnet.classify(data[i])
		if data[i].evs[0] == 1:
			if outcome == 0:
				color = [[0,1,0]]
			else:
				color = [[0,0.5,0]]
		else:
			if outcome == 1:
				color = [[1,0,0]]
			else:
				color = [[0.5,0,0]]

		plt.scatter(data[i].inputs[0], data[i].inputs[1], c=color)
	plt.show()

def train_network():
	nnet = NeuralNetwork([2, 3, 2])

	numData = 400
	sampleSize = 50
	data = generate_data(numData)
	# numData = len(data)
	while True:
		correctPoints, _, _ = nnet.test_points(data)
		print(nnet.cost_average(data))
		print(f"{correctPoints}/{numData}")

		# if correctPoints >= (numData-5):
		# graph_data(data, nnet)

		graph_data(data, nnet)
		for i in range(500):
			# nnet.learn(r.sample(data, sampleSize), .1)
			nnet.learn_with_derivatives(r.sample(data, sampleSize), .25)
		break
	nwrap.save_network(nnet)

def	gradients_test():
	oldNet = NeuralNetwork([2,3,2])
	newNet = NeuralNetwork([2,3,2])

	numData = 200
	sampleSize = 200
	data = generate_data(numData)

	for layerIndex in range(len(oldNet.layers)):
		newNet.layers[layerIndex] = copy.deepcopy(oldNet.layers[layerIndex])

	# while True:
	# 	for i in range(500):
	# 		oldNet.learn(r.sample(data, 50), 0.5)
	# 		# newNet.learn_with_derivatives(data, 0.35)
	# 	correctPoints, _, _ = oldNet.test_points(data)
	# 	if correctPoints >= (numData-15):
	# 		graph_data(data, oldNet)
	# 		break
	# for layerIndex in range(len(oldNet.layers)):
	# 	newNet.layers[layerIndex] = copy.deepcopy(oldNet.layers[layerIndex])

	for i in range(10):
		oldNet.learn(data, 0.35)
		newNet.learn_with_derivatives(data, 0.35)
	for layerIndex in range(len(oldNet.layers)):
		newNet.layers[layerIndex] = copy.deepcopy(oldNet.layers[layerIndex])
	for i in range(150):
		oldNet.learn(data, 0.35)
		newNet.learn_with_derivatives(r.sample(data, 50), 0.70)
	print("GRADIENTS")
	print(oldNet.layers[0].costGradientWeights)
	print(newNet.layers[0].costGradientWeights)
	print(oldNet.layers[1].costGradientWeights)
	print(newNet.layers[1].costGradientWeights)

	print(oldNet.layers[0].costGradientBiases)
	print(newNet.layers[0].costGradientBiases)
	print(oldNet.layers[1].costGradientBiases)
	print(newNet.layers[1].costGradientBiases)

	print()
	print("VALUES")
	print(oldNet.layers[0].weights)
	print(newNet.layers[0].weights)
	print()
	print(oldNet.layers[1].weights)
	print(newNet.layers[1].weights)
	print()

	print(oldNet.layers[0].biases)
	print(newNet.layers[0].biases)
	print()
	print(oldNet.layers[1].biases)
	print(newNet.layers[1].biases)
	graph_data(data, oldNet)

	# while True:
	# 	for i in range(500):
	# 		# oldNet.learn(r.sample(data, 50), 0.5)
	# 		newNet.learn_with_derivatives(data, 1)
	# 	correctPoints, _, _ = newNet.test_points(data)
	# 	print(newNet.cost_average(data))
	# 	print(f"{correctPoints}/{numData}")
	# 	# graph_data(data, newNet)



def main():

	# train_network()
	# gradients_test()

	nnet = nwrap.create_network([2,3,2])
	numData = 400
	sampleSize = 50
	data = generate_data(numData)
	nwrap.train_network(nnet, 0.2, data, sampleSize, 95)
	print("done training")
	nwrap.save_network(nnet)
	print("done saving")
	name = input("type name of network to import")
	nnet2 = nwrap.create_network([1,1], name)
	graph_data(data, nnet2)



if __name__ == '__main__':
	main()
