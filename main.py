import numpy as numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


trainingInputs = numpy.array([[0, 0, 1],
                              [1, 1, 1],
                              [1, 0, 1],
                              [0, 1, 1]])

trainingOutputs = numpy.array([[0, 1, 1, 0]]).T

numpy.random.seed(1)
synapticWeights = 2 * numpy.random.random((3, 1)) - 1

# back propagation method
for i in range(20000):
    inputLayer = trainingInputs
    outputs = sigmoid(numpy.dot(inputLayer, synapticWeights))

    err = trainingOutputs - outputs
    adjustments = numpy.dot(inputLayer.T, err * (outputs * (1 - outputs)))

    synapticWeights += adjustments

    print("Generation ", i, " weights:")
    print(synapticWeights)

print("Weights after training:")
print(synapticWeights)
