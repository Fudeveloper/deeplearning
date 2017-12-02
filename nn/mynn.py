import numpy


# 双曲线函数
def tanh(x):
    return numpy.tanh(x)


# 一阶导数
def tanh_deriv(x):
    return 1.0 - (tanh(x) * tanh(x))


# 逻辑函数
def logistic(x):
    return 1 / (1 + numpy.exp(-x))


# 逻辑函数一阶导数
def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


# 神经网络类
class NeuralNetwork:
    # params： 神经网络层数，激活方法(非线性转化过程)
    # 神经网络层数example：(10,10,3)
    def __init__(self, layers, activation="tanh"):
        if activation == "tanh":
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation == "logistic":
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        # 初始化时，增加随机的权重
        # 以隐藏层为基准（中间向两端）加权重
        self.weights = []
        for i in range(1, len(layers)-1):
            # # i-1层和i层之间的权重
            self.weights.append((2 * numpy.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            # # i层和i+1层之间的权重
            self.weights.append((2 * numpy.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    # epochs=10000 最多循环次数
    def fit(self, X, y, leraning_rate=0.2, epochs=10000):
        # 判断x至少为2维
        X = numpy.atleast_2d(X)
        # X.shape[0]:X矩阵行数
        #  X.shape[1]:X矩阵列数
        # 赋值偏向
        X = numpy.atleast_2d(X)
        temp = numpy.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        # print(X)
        y = numpy.array(y)

        for k in range(epochs):
            # 从X矩阵中取随机一行数据
            i = numpy.random.randint(X.shape[0])
            a = [X[i]]

            # 正向更新
            for r in range(len(self.weights)):
                # a.append(self.activation(numpy.dot(a[1], self.weights[1])))
                a.append(self.activation(numpy.dot(a[r], self.weights[r])))
            # 计算误差（真实值-预测值）
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # 反向更新
            # 最后一层到第0层（除去输出层和输入层），每次往回退1层
            for i in range(len(a) - 2, 0, -1):
                # 更新隐藏层
                deltas.append(deltas[-1].dot(self.weights[i].T) * self.activation_deriv(a[i]))

            deltas.reverse()
            # 更新权重
            for i in range(len(self.weights)):
                layer = numpy.atleast_2d(a[i])
                delta = numpy.atleast_2d(deltas[i])
                self.weights[i] += leraning_rate * layer.T.dot(delta)

    def predict(self, x):
        # 类似于一次正向更新
        x = numpy.array(x)
        # 加入偏向
        temp = numpy.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for i in range(0, len(self.weights)):
            a = self.activation(numpy.dot(a, self.weights[i]))

        return a
