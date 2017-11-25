import random
import math
import csv
import operator
import pandas


# 装载数据集
# 将所有数据集分为两部分：训练集和测试集
def load_dataset(filename, split, traning_set=[], test_set=[]):
    with open(filename) as f:
        for line in f:
            data = line.split(',')
            for i in range(len(data) - 1):
                data[i] = float(data[i])
            if random.random() < split:
                traning_set.append(data)
            else:
                test_set.append(data)
                # print(len(traning_set))
                # print(len(test_set))


# 根据传入的实例和维度，计算它们的距离
def calc_distance(instance1, instance2, dimen):
    distance = 0
    for i in range(dimen):
        distance += math.pow(instance1[i] - instance2[i], 2)
    return math.sqrt(distance)


# 根据传入的k值和测试实例，获取相应个数的邻近点（k一般为奇数）
def get_neighbors(train_set, test_instance, k):
    distance_set = []
    # test_instance = test_instance.split(',')
    dimen = len(test_instance) - 1
    # print(dimen)

    # 计算训练集中的所有数据和测试实例的距离,并将它们标记存储(存储为元组)
    for i in range(len(train_set)):
        distance = calc_distance(train_set[i], test_instance, dimen)
        distance_set.append((distance, train_set[i]))
    distance_set.sort(key=operator.itemgetter(0))
    # print(distance_set)
    # 根据k值，返回相应个数的最临近点
    neighbors = []
    for x in range(k):
        neighbors.append(distance_set[x][1])
    # print(neighbors)
    return neighbors


# 投票选取归属种类
def get_response(neighbors):
    vote = {}
    # 遍历neighbors，在字典中，如果有对应键，则其投票数加1，如果没有对应键，则其投票数设为1
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in vote:
            vote[response] += 1
        else:
            vote[response] = 1
    # 将字典按投票数量排序
    sorted_vote = sorted(vote.items(), key=lambda x: x[1], reverse=True)
    return sorted_vote[0][0]
    #
    # print(vote)
    # print(sorted_vote)


# 精确度评价
def get_accuary(test_set, preditons):
    correct = 0
    all = len(test_set)
    for i in range(all):
        if test_set[i][-1] == preditons[i]:
            correct += 1
    return (correct / float(all)) * 100


def main():
    traning_set = []
    test_set = []
    k = 3
    load_dataset('irisdata.csv', 0.6, traning_set, test_set)
    preditons = []
    for i in range(len(test_set)):

        neighbors = get_neighbors(traning_set, test_set[i], k)
        result = get_response(neighbors)
        preditons.append(result)

    print(get_accuary(test_set, preditons))


main()
