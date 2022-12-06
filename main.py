import numpy as np
import platform

from DataPoint import DataPoint
from NeuralNetwork import NeuralNetwork
import NNetWrapper

if platform.system() == "Windows": from mnist.loader import MNIST
else: from mnist import MNIST


# TODO: make code that is nice to run for project presentation
# TODO: import and process custom images?


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
    inputLoop = True
    print("Type (Q) at any point to quit")
    userInput = input("Type (N) to create a network or (I) to import a network\n > ").lower().strip()
    while inputLoop:
        if userInput == "n":
            userLayers = input("Type the layer input string (see readme)\n > ")
            userLayers = [int(x) for x in userLayers.split(',')]
            nnet = NNetWrapper.create_network(userLayers)
            inputLoop = False
        elif userInput == "i":
            userFileName = input("Type the name of the saved network to import\n > ")
            nnet = NNetWrapper.import_network(userFileName)
            inputLoop = False
        elif userInput == "q":
            print("Exiting")
            return 0

    inputLoop = True
    while inputLoop:
        userInput = input("Type (C) to identify a custom image\n > ").lower().strip()
        if userInput == "c":
            userImageName = input("Press enter once an image has been saved to the images folder\n > ")
            if not userImageName:
                userImageName = "test"
            digitGuess = NNetWrapper.classify_digit(userImageName, nnet)
            print(f"Network Classification: {digitGuess}")
        elif userInput == "q":
            print("Exiting")
            return 0





if __name__ == "__main__":
    main()