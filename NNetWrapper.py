import copy
import cv2 as cv
import numpy as np
import pickle
import random as r

from DataPoint import DataPoint
from NeuralNetwork import NeuralNetwork

# create and return a network with the specified layers
# when creating a network it picks an ID that isn't yet present in the networks folder
def create_network(layers):
    return NeuralNetwork(layers)

# import a network from the networks folder given a filename
def import_network(networkName):
    with open(f"networks/{networkName}.pickle", 'rb') as inputFile:
        nnet = pickle.load(inputFile)
    return nnet

# save the network to a file, the network ID gives each network a unique file name
def save_network(nnet: NeuralNetwork):
    with open(f"networks/network{nnet.networkID}.pickle", 'wb') as outputFile:
        pickle.dump(nnet, outputFile, pickle.HIGHEST_PROTOCOL)

# given data and training parameters, train the network
def train_network(nnet: NeuralNetwork, learnRate, trainingData: list[DataPoint], withPrint=False, batchSize=100, cutoffPercent=90, cycles=9999999999, itersPerAccuracyCheck=10):
    numData = len(trainingData)
    epochCount = 0
    totalTrainings = 0
    epochProgress = 0
    correct = 0
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
    return correct

# take a 28x28 image and center the number within by pixel mass
# also invert the pixel values if the image is using 0 for foreground (black on white)
# instead of background (white on black)
def preprocess_image(imageName):
    image = cv.imread(f"images/{imageName}.png", 1)

    ## check top left corner brightness to guess background alpha
    if image[0][0][0] > 32:
        ## invert the colors if necessary
        outImage = copy.deepcopy(image)
        for row in range(28):
            for col in range(28):
                inverse = 255 - image[row][col]
                outImage[row][col] = inverse

    ## find center of mass
    xbar = 0
    ybar = 0
    pixelMass = 0
    for row in range(28):
        for col in range(28):
            xbar += outImage[row][col][0] * (14 - col)
            ybar += outImage[row][col][0] * (14 - row)
            pixelMass += outImage[row][col][0]
    xbar = np.trunc(xbar/pixelMass)
    ybar = np.trunc(ybar/pixelMass)

    ## center pixels using center of mass
    outImage = cv.warpAffine(outImage, np.float32([[1, 0, xbar], [0, 1, ybar]]), (28, 28))

    ## save
    cv.imwrite(f"images/{imageName}-processed.png", outImage)

# take a filename, preprocess it and run it through and classify it
def classify_digit(imageName, nnet: NeuralNetwork):
    preprocess_image(imageName)
    image = cv.imread(f"images/{imageName}-processed.png")

    # generate DataPoint object from image data
    imageData = [0] * 784
    for row in range(28):
        for col in range(28):
            imageData[col + 28 * row] = image[row][col][0]
    point = DataPoint(imageData, [-1] * 10)

    digitGuess = nnet.classify(point) # run the network
    return digitGuess
