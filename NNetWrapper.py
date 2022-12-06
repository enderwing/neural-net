import copy
import cv2 as cv
import numpy as np
import pickle
import random as r

from DataPoint import DataPoint
from NeuralNetwork import NeuralNetwork

def create_network(layers):
    return NeuralNetwork(layers)

def import_network(filePath):
    with open(filePath, 'rb') as inputFile:
        nnet = pickle.load(inputFile)
    return nnet

def save_network(nnet: NeuralNetwork):
    with open(f"networks/network{nnet.networkID}.pickle", 'wb') as outputFile:
        pickle.dump(nnet, outputFile, pickle.HIGHEST_PROTOCOL)

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

def preprocess_image(imageName):
    image = cv.imread(f"images/{imageName}.png", 1)

    ## invert the colors
    outImage = copy.deepcopy(image)
    for row in range(28):
        for col in range(28):
            inverse = 255 - image[row][col]
            outImage[row][col] = inverse
            # if row == 27:
            #     outImage[row][col] = [255, 255, 255]
            # if col == 27:
            #     outImage[row][col] = [127, 127, 127]

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
    print(f"x-offset: {xbar:.1}")
    print(f"y-offset: {ybar:.1}")

    ## center pixels using center of mass
    outImage = cv.warpAffine(outImage, np.float32([[1, 0, xbar], [0, 1, ybar]]), (28, 28))

    cv.imwrite(f"images/{imageName}-processed.png", outImage)