import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from mynn import NeuralNetwork
from sklearn.cross_validation import train_test_split
start = time.time()
# 装载数据集
digits = load_digits()
x = digits.data
y = digits.target
x -= x.min()
x /= x.max()

# 实例化一个nn
# 64：图片像素点8*8，
# 隐藏层：100：一般比输入层多，数量灵活
# 10：需要区分的类别个数
nn = NeuralNetwork([64, 100, 10], 'logistic')

# 拆分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 将标签转化为矩阵
lables_train = LabelBinarizer().fit_transform(y_train)
lables_test = LabelBinarizer().fit_transform(y_test)
# print(lables_test)

# 建模


nn.fit(x_train, lables_train, epochs=30000, leraning_rate=0.3)

predictions = []
for i in range(x_test.shape[0]):
    # 预测
    array = nn.predict(x_test[i])
    # 取出概率最大的对应索引
    predictions.append(np.argmax(array))

# 评价准确率
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
end = time.time()

print(end-start)