import pickle
import random as r

from DataPoint import DataPoint
from NeuralNetwork import NeuralNetwork

def create_network(layers, filePath=None):
    if filePath:
        with open(filePath, 'rb') as inputFile:
            nnet = pickle.load(inputFile)
        return nnet
    else:
        return NeuralNetwork(layers)

def save_network(nnet: NeuralNetwork):
    with open(f"networks/network{nnet.networkID}.pickle", 'wb') as outputFile:
        pickle.dump(nnet, outputFile, pickle.HIGHEST_PROTOCOL)

def train_network(nnet: NeuralNetwork, learnRate, trainingData: list[DataPoint], withPrint=False, batchSize=100, cutoffPercent=90, cycles=9999999999, itersPerAccuracyCheck=10):
    numData = len(trainingData)
    epochCount = 0
    totalTrainings = 0
    epochProgress = 0
    while totalTrainings < cycles:
        # cost, correct, _, _ = nnet.points_info(trainingData)
        cost = nnet.cost_average(trainingData)
        correct, _, _ = nnet.test_points(trainingData)
        if withPrint:
            print(cost)
            print(f"{correct}/{numData}")
        if correct >= numData * cutoffPercent/100:
            break
        for i in range(itersPerAccuracyCheck):
            sampleData = trainingData[epochProgress:epochProgress + batchSize]
            epochProgress += batchSize
            if withPrint:
                print(f"epoch #{epochCount} progress: {epochProgress/numData * 100:2.2f}")
            if epochProgress + batchSize > len(trainingData):
                r.shuffle(trainingData)
                epochProgress = 0
                epochCount += 1
            nnet.learn_with_derivatives(sampleData, learnRate)
            totalTrainings += 1
