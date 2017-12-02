import numpy
from sklearn import datasets, linear_model

# 装载数据（数据矩阵）
data = numpy.genfromtxt(r'Delivery_Dummy.csv', delimiter=',')

x = data[:, 0: -1]
y = data[:, -1]

lr = linear_model.LinearRegression()
lr.fit(x, y)
# 模型中的参数预测
print(lr.coef_)

# 模型中的截距
print(lr.intercept_)
print(lr.predict([[102, 6, 0, 1, 0]])[0])
