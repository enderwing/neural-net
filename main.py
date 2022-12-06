import numpy as np
import platform
import random as r

from DataPoint import DataPoint
import NNetWrapper

if platform.system() == "Windows": from mnist.loader import MNIST
else: from mnist import MNIST


# imports data from the MNIST data set for training and testing
def import_data(numData, numTestData, mndata):
    print("reading data")
    trainImages, trainLabels = mndata.load_training()
    testImages, testLabels = mndata.load_testing()

    print("preparing data")
    trainingDataPoints = []
    testDataPoints = []

    # turn data from idx3 format into DataPoint objects for passing to the network
    for i in range(numData):
        evs = [0] * 11
        evs[trainLabels[i]] = 1
        evs[10] = trainLabels[i]
        trainingDataPoints.append(DataPoint(list(np.divide(trainImages[i], 255)), evs))
    for i in range(numTestData):
        evs = [0] * 11
        evs[testLabels[i]] = 1
        evs[10] = testLabels[i]
        testDataPoints.append(DataPoint(list(np.divide(testImages[i], 255)), evs))

    return trainingDataPoints, testDataPoints, testImages, testLabels

# main user interaction
def main():
    mndata = MNIST('data')
    trainingData, testData, testImages, testLabels = import_data(0, 10000, mndata)
    print("done")

    # create or import a network
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

    # classify a custom image or pick an image from the MNIST test set
    inputLoop = True
    while inputLoop:
        userInput = input("Type (C) to identify a custom image or (T) to use a random image from the test data\n > ").lower().strip()
        if userInput == "c":
            userImageName = input("Type the filename (w/o file extension) after saving it to the images folder\n > ")
            if not userImageName:
                userImageName = "test"
            digitGuess = NNetWrapper.classify_digit(userImageName, nnet)
            print(f"Network Classification: {digitGuess}")
        elif userInput == "t":
            displayTestLoop = True
            while displayTestLoop:
                randIndex = r.randint(0, len(testData))
                print(mndata.display(testImages[randIndex]))
                print(f"Expected Value: {testData[randIndex].evs[10]}")
                print(f"Network Classification: {nnet.classify(testData[randIndex])}\n")
                userInput = input("Enter to display another or (Q) to go back").strip().lower()
                if userInput == "q":
                    displayTestLoop = False
        elif userInput == "q":
            print("Exiting")
            return 0


if __name__ == "__main__":
    main()