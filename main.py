import numpy as np
import platform

from DataPoint import DataPoint
from NeuralNetwork import NeuralNetwork
import NNetWrapper

if platform.system() == "Windows": from mnist.loader import MNIST
else: from mnist import MNIST


#TODO: make code that is nice to run for project presentation
#TODO: import and process custom images?


def import_data(numData, numTestData):
    mndata = MNIST('data')

    print("reading data")
    trainImages, trainLabels = mndata.load_training()
    testImages, testLabels = mndata.load_testing()

    print("preparing data")
    trainingData = []
    testData = []

    for i in range(numData):
        evs = [0] * 11
        evs[trainLabels[i]] = 1
        evs[10] = trainLabels[i]
        trainingData.append(DataPoint(list(np.divide(trainImages[i], 255)), evs))

    for i in range(numTestData):
        evs = [0] * 11
        evs[testLabels[i]] = 1
        evs[10] = testLabels[i]
        testData.append(DataPoint(list(np.divide(testImages[i], 255)), evs))

    return trainingData, testData


def main():

    userInput = input("Type (N) to create a network or (I) to import a network\n > ").lower().strip()
    if userInput == "n":
        userLayers = input("Type the layer input string (see readme)\n > ")
        userLayers = [int(x) for x in userLayers.split(',')]
        nnet = NNetWrapper.create_network(userLayers)
    elif userInput == "i":
        userFileName = input("Type the file path with the network file to import\n > ")
        nnet = NNetWrapper.import_network(userFileName)

    userInput = input("Type (C) to identify a custom image\n > ").lower().strip()
    if userInput == "c":
        userImageName = input("Type the name of the image (which must be stored in the images folder)\n > ")



if __name__ == "__main__":
    main()