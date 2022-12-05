import platform

if platform.system() == "Windows":
    from mnist.loader import MNIST
else:
    from mnist import MNIST
import time as t
import random as r

from NeuralNetwork import NeuralNetwork
from DataPoint import DataPoint

mndata = MNIST('data')

trainImages, trainLabels = mndata.load_training()

nnet = NeuralNetwork([784, 10])
data = []
numData = 10
sampleSize = 10
for i in range(10):
    evs = [0] * 10
    evs[trainLabels[i]] = 1
    data.append(DataPoint(trainImages[i], evs))

while True:
    print(nnet.cost_average(data))
    print(f"{nnet.test_points(data)}/{numData}")
    # for layer in nnet.layers:
    #     print(layer.weights)
    #     print(layer.biases)
    nnet.learn(r.sample(data, sampleSize), 0.25)
