from mnist import MNIST
import time as t

mndata = MNIST('data')

trainDigits, trainLables = mndata.load_training()

print(len(trainImages[0]))