import numpy as np
import math


# 计算两个向量的相关度
def computer_correlation(x, y):
    x_aver = np.mean(x)
    y_aver = np.mean(y)
    SSR = 0
    var_x = 0
    var_y = 0

    for i in range(len(x)):
        diff_x_x_aver = x[i] - x_aver
        diff_y_y_aver = y[i] - y_aver
        SSR += diff_x_x_aver * diff_y_y_aver
        var_x += diff_x_x_aver ** 2
        var_y += diff_y_y_aver ** 2
    SST = np.sqrt(var_y * var_x)
    return SSR / SST


x = [1, 3, 8, 7, 9]
y = [10, 12, 24, 21, 34]

#
r = computer_correlation(x, y)
# r平方：决定系数
R = r ** 2
print(R)
