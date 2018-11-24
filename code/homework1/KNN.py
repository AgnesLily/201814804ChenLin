import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
import heapq
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
file_path = "../../data"


def process_data(path):
    train_x = []
    train_y = []
    files = os.listdir(path)
    # index = 0
    label_index = 0
    for file in files:
        labels = os.listdir(path)
        file_list = os.listdir(path + "/" + file)
        for vsm in file_list:
            temp = []
            doc = open(path + "/" + file + "/" + vsm, 'r', encoding="ISO-8859-1")
            for row in doc:
                temp.append(float(row))
            train_x.append(temp)
            train_y.append(labels[label_index])
            # index += 1
        label_index += 1
    return train_x, train_y


# Euclidean distance
def cal_euclidean(train_item, test_item):
    vec_train = np.array(train_item)
    vec_test = np.array(test_item)
    vec_sub = vec_train - vec_test
    vec_sub = vec_sub**2
    distance = np.sum(vec_sub)
    distance = np.sqrt(distance)
    # sum = 0
    # for i in range(len(vec_train)):
    #     sum += (vec_train[i] - vec_test[i]) * (vec_train[i] - vec_test[i])
    # distance = math.sqrt(sum)
    return distance


# cosine distance
def cal_cosine(train_item, test_item):
    vec_train = np.array(train_item)
    vec_test = np.array(test_item)
    distance = np.dot(vec_train, vec_test) / (np.linalg.norm(vec_train) * (np.linalg.norm(vec_test)))
    return distance


def KNN(path, K):
    train_samples, train_labels = process_data(path)
    train_x, test_x, train_y, test_y = train_test_split(train_samples, train_labels, test_size=0.2, random_state=1)
    # print(len(train_x), len(train_y), len(test_x), len(test_y))
    labels_true = []
    labels_pre = []
    # 遍历测试集中的每个元素
    # for indexl in range(len(test_x)):
    #     test_item = test_x[indexl]
    #     labels_true.append(test_y[indexl])
    #     # test_index_true = test_y[indexl]
    #     result = []
    #     # 遍历训练集中的每个元素，找出距离最近的k个元素的类别
    #     for index in range(len(train_x)):
    #         train_item = train_x[index]
    #         # dis = cal_euclidean(train_item, test_item)
    #         dis = euclidean_distances(np.array(train_item), np.array(test_item))
    #         print(dis)
    #         # dis = cal_cosine(train_item, test_item)
    #         result.append(dis)
    #     print("done")
    #     min_distances = map(result.index, heapq.nsmallest(K, result))
    #     print("done")
    #     # print(list(min_distances))
    #     labels = []
    #     for item in list(min_distances):
    #         pre_test_label = train_y[item]
    #         labels.append(pre_test_label)
    #     label_counts = Counter(labels)
    #     top_one = label_counts.most_common(1)
    #     labels_pre.append(top_one[0][0])
    #     print(indexl)
    #     print(top_one[0][0])
    # dis_matrix = euclidean_distances(test_x, train_x)
    dis_matrix = cosine_distances(test_x, train_x)

    # print(len(dis_matrix))
    for index in range(len(dis_matrix)):
        labels_true.append(test_y[index])
        dis_array = dis_matrix[index]
        dis_array = dis_array.tolist()
        min_distances = map(dis_array.index, heapq.nsmallest(K, dis_array))
        labels = []
        for item in list(min_distances):
            pre_test_label = train_y[item]
            labels.append(pre_test_label)
        label_counts = Counter(labels)
        top_one = label_counts.most_common(1)
        labels_pre.append(top_one[0][0])
        # print(top_one[0][0])
    # 计算分类准确率
    right = 0
    for i in range(len(labels_pre)):
        if labels_pre[i] == labels_true[i]:
            right += 1
    acc = right/len(labels_pre)
    print("When k is " + str(K) + " ,the acc of KNN classifier with cosine distance is :" + str(acc))


# 计算acc
if __name__ == "__main__":
    for i in range(1, 61):
        KNN(file_path, i)
