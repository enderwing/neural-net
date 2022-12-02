from DataPoint import DataPoint
from NeuralNetwork import NeuralNetwork

import math as m
import random as r
import time as t
import matplotlib.pyplot as plt
import numpy as np


def generate_data(amount):
	data = []
	for i in range(amount):
		point = [r.random() * 10 - 5, r.random() * 10 - 5]
		check = 1 if m.sqrt(point[0]**2 + point[1]**2) < 2.52 else 0
		# check = 1 if point[1] < 2.5 else 0
		data.append(DataPoint(point, [check, 0 if check else 1]))
	return data

def graph_data(data: list[DataPoint], nnet: NeuralNetwork):
	theta = np.arange(0, np.pi*2, 0.01)
	plt.plot(2.52 * np.cos(theta), 2.52 * np.sin(theta))
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

def main():
	nnet = NeuralNetwork([2, 3, 2])

	numData = 50
	data = generate_data(numData)
	while True:
		print(nnet.cost_average(data))
		print(f"{nnet.test_points(data)}/{numData}")
		for layer in nnet.layers:
			print(layer.weights)
			print(layer.biases)
		graph_data(data, nnet)
		for i in range(1000):
			nnet.learn(data, 0.15)



if __name__ == '__main__':
	main()
