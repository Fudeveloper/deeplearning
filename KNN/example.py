from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
# 从数据集中读取矩阵
iris = datasets.load_iris()

# print(iris)
# print(iris.target)
# 建模（特征值（二维矩阵），结果集（一维数组））
knn.fit(iris.data, iris.target)

# 预测新实例的类别
predict_result = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print(predict_result)
