import mynn
import numpy

nn = mynn.NeuralNetwork([2, 2, 1])
X = numpy.array([[0, 1], [0, 0], [1, 1], [1, 0]])
Y = numpy.array([1, 0, 0, 1])

nn.fit(X, Y)
for i in [[0, 1], [0, 0], [1, 1], [1, 0]]:
    print(nn.predict(i))
