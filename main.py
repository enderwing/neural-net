from NeuralNetwork import NeuralNetwork

import math as m
import matplotlib.pyplot as plt

def generateFruits(amount):
	fruits = [[m.randint(1,10), m.randint(1,10)] for i in range(amount)]

def main():
	nnet = NeuralNetwork([2, 3, 2])
	nnet.calculate_outputs([1, 2])


if __name__ == '__main__':
	main()
