# 决策树
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing


allElectronics = open('AllElectronics.csv', 'r')
reader = csv.reader(allElectronics)
headers = next(reader)
# print(headers)

feature_list = []
class_list = []

for row in reader:
    item_nums = len(row) - 1
    temp_list = {}
    class_list.append(row[item_nums])
    for i in range(1, item_nums):
        temp_list[headers[i]] = row[i]
    feature_list.append(temp_list)
#
# print(feature_list)
# print(class_list)

# 将特征值转换为矩阵
vec = DictVectorizer()
feature_bin = vec.fit_transform(feature_list).toarray()
# print(dum)

# 将标记值转换为矩阵
lb = preprocessing.LabelBinarizer()
class_bin = lb.fit_transform(class_list)
# print(class_bin)

# 选择信息熵算法选择决策树根节点
dtc = tree.DecisionTreeClassifier(criterion='entropy')

# 建模
res = dtc.fit(feature_bin, class_bin)

# 通过graphviz画出决策树
with open('result.dot', 'w') as f:
    f = tree.export_graphviz(dtc, feature_names=vec.get_feature_names(), out_file=f)
# cmd中运行下面命令将dot文件转为pdf
# D:/graphviz/bin/dot -Tpdf input.dot -o output.pdf

# 取源矩阵数据第一行，修改点内容，预测该数据结果
new_man = feature_bin[0:1]
print(new_man)

new_man[0][0] = 1
new_man[0][2] = 0

predict_result = dtc.predict(new_man)
print(predict_result)
