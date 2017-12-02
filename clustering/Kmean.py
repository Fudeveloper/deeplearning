import numpy as np


def kmeans(x, k, loop_times=1000):
    # 构造矩阵(比数据源多一列，用来存储分组信息)
    lines, columns = np.shape(x)
    data_set = np.zeros([lines, columns + 1])
    data_set[:, 0:-1] = x
    # print(data_set)
    # 选取k个随机中心点
    centroids = data_set[np.random.randint(lines, size=k)]
    # print(centroids[:, -1])
    # 初始化分类
    centroids[:, -1] = range(1, k + 1)

    iterations = 0
    old_centroids = None

    while not should_stop(old_centroids, centroids, iterations, loop_times):
        old_centroids = np.copy(centroids)
        iterations += 1
        update_lables(data_set, centroids)
        centroids = get_centroids(data_set, k)
    return data_set


def should_stop(old_centroids, centroids, iterations, loop_times):
    if iterations > loop_times:
        return True
    return np.array_equal(old_centroids, centroids)


def update_lables(data_set, centroids):
    lines, columns = np.shape(data_set)
    for i in range(lines):
        print(data_set[i, : -1])
        # data_set[i, -1] = get_lable_from_cloest_centriods(data_set[i, : -1], centroids)


# 从最近的中心点得到分类标签
def get_lable_from_cloest_centriods(data_row, centroids):
    lines, columns = np.shape(centroids)
    print(centroids)
    lable = centroids[0][-1]
    # print(lable)
    # 计算距离最近的中心点
    # 计算第一个点与中心点的距离
    min_dist = np.linalg.norm(data_row - centroids[0][0:-1])
    # 计算每个点与中心点的距离
    for i in range(1, lines):
        dist = np.linalg.norm(data_row - centroids[i][0:-1])
        if dist < min_dist:
            min_dist = dist
            lable = centroids[i][0:-1]
    return lable


def get_centroids(data_set, k):
    lines, columns = np.shape(data_set)
    print(lines,columns)
    result = np.zeros([k, columns])
    # 计算所有标签相同的点的均值
    for i in range(1, k + 1):
        # 找到标签相同的所有点
        one_cluster = data_set[data_set[:, -1] == i][-1]
        print(one_cluster)
        # 求这些点的均值（行）
        result[i - 1, :-1] = np.mean(one_cluster, axis=0)
        result[i - 1, -1] = i
    return result


k = 2
# x = [, , , ]
x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
test_x = np.vstack((x1, x2, x3, x4))
result = kmeans(test_x, k, 10)

# print(result)
