from sklearn import svm

# 特征向量值矩阵
x = [[2, 0], [1, 1], [2, 3]]

# 类标签
class_lables = [0, 0, 1]

clf = svm.SVC(kernel='linear')

clf.fit(x, class_lables)

print(clf.support_vectors_)

print(clf.predict([[3, 3]]))  # [1]
