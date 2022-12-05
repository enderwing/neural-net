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
numData = 1000
sampleSize = 100
for i in range(numData):
    evs = [0] * 11
    evs[trainLabels[i]] = 1
    evs[10] = trainLabels[i]
    data.append(DataPoint(trainImages[i], evs))


while True:
    print(nnet.cost_average(data))
    correct, _, _ = nnet.check_digits(data)
    print(f"{correct}/{numData}")
    # for layer in nnet.layers:
    #     print(layer.weights)
    #     print(layer.biases)
    for i in range(100):
        sampleData = r.sample(data, sampleSize)
        nnet.learn_with_derivatives(sampleData, 1)
    # count = 0
    # for i in range(numData):
    #     print(mndata.display(trainImages[i]))
    #     print(data[i].evs)
    #     print(nnet.classify(data[i]))
    #     count += 1
    #     if count > 0:
    #         input()
    #         count = 0

