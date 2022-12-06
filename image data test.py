import pickle
import platform
if platform.system() == "Windows":
    from mnist.loader import MNIST
else:
    from mnist import MNIST

import time as t
import random as r
import numpy as np

from NeuralNetwork import NeuralNetwork
import NNetWrapper as nwrap
from DataPoint import DataPoint


mndata = MNIST('data')

print("reading data")
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

print("preparing data")
nnet = NeuralNetwork([784, 50, 10])
trainingData = []
numData = 60000
batchSize = 250
for i in range(numData):
    evs = [0] * 11
    evs[trainLabels[i]] = 1
    evs[10] = trainLabels[i]
    trainingData.append(DataPoint(list(np.divide(trainImages[i], 255)), evs))

print("beginning network training...")
epochCount = 1
epochProgress = 0
try:
    trainNetwork = True
    while trainNetwork:
        # cost, correct, _, _ = nnet.points_info(trainingData)
        # print(cost)
        # print(f"{correct}/{numData}")
        # if correct == numData:
        #     break
        for i in range(240):
            sampleData = trainingData[epochProgress:epochProgress + batchSize]
            epochProgress += batchSize
            print(f"epoch #{epochCount} progress: {epochProgress/numData * 100:2.2f}")
            if epochProgress + batchSize > len(trainingData):
                r.shuffle(trainingData)
                epochProgress = 0
                epochCount += 1
            nnet.learn_with_derivatives(sampleData, .06)
        trainNetwork = False
except KeyboardInterrupt:
    pass

print("running test")
cost, correct, _, _ = nnet.points_info(trainingData)
print(cost)
print(f"{correct}/{numData}")

# save the network state
nwrap.save_network(nnet)


print("done dsaving")
input()
# prepare test data
print("preparing test data")
testData = []
for i in range(len(testLabels)):
    evs = [0] * 11
    evs[testLabels[i]] = 1
    evs[10] = testLabels[i]
    testData.append(DataPoint(list(np.divide(testImages[i], 255)), evs))
passed = []
failed = []
print("testing test data !")
for i, point in enumerate(testData):
    if nnet.check_digit(point):
        passed.append([point, i])
    else:
        failed.append([point, i])

print("~~~ TEST DATA RESULTS ~~~")
print(f"CHRIS!    Passed num: {len(passed)}")
print(f"HI!       Failed num: {len(failed)}")
print("press enter to see failed tests")
while True:
    point = failed[r.randint(0,len(failed)-1)]
    print(point[0].evs)
    print(mndata.display(testImages[point[1]]))
    print(nnet.classify(point[0]))
    input()


    # count = 0
    # for i in range(numData):
    #     print(mndata.display(trainImages[i]))
    #     print(data[i].evs)
    #     print(nnet.classify(data[i]))
    #     count += 1
    #     if count > 0:
    #         input()
    #         count = 0

