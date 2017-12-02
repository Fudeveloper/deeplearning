import numpy as np
import random


def gen_data(numpoints, bias, variance):
    # features
    x = np.zeros(shape=(numpoints, 2))
    # class lables
    y = np.zeros(shape=numpoints)
    for i in range(numpoints):
        x[i][0] = 1
        x[i][1] = i
        y[i] = random.uniform(0, 1) * variance + (i + bias)
    return x, y


# x:实例矩阵
# y:标签向量
# theta:需要学习的参数
# alpha:学习率
# m:实例数
# num_iterations:学习(更新)次数

def gradient_descent(x, y, theta, alpha, m, num_iterations):
    # 转置
    x_train = x.transpose()

    for i in range(num_iterations):
        # 更新法则
        # 期望值
        hypothesis = np.dot(x, theta)
        # print(hypothesis)
        # 计算期望值和实际值的差
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("次数：{}，cost：{}".format(i, cost))
        gradient = np.dot(x_train, loss) / m
        theta = theta - alpha * gradient
    return theta


x, y = gen_data(100, 25, 10)

m, n = np.shape(x)
n_y = np.shape(y)

# print("m:" + str(m) + " n:" + str(n) + " n_y:" + str(n_y))

numIterations = 10000
alpha = 0.0005
theta = np.ones(n)
theta = gradient_descent(x, y, theta, alpha, m, numIterations)
print(theta)
# print(y)
