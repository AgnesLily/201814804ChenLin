import os
import math
import numpy as np
from sklearn.model_selection import train_test_split

file_path = "../../dataset2"


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


# 划分数据集
train_samples, train_labels = process_data(file_path)
train_x, test_x, train_y, test_y = train_test_split(train_samples, train_labels, test_size=0.2, random_state=1)


#分词函数
# def extract_doc():


#计算类别的文档数
def cal_num():
    nums = []
    return nums

doc_nums = cal_num()


# def train_naive_bayes(category, docs):
#     voc = extract_doc()
#     docs_num = 18828
#     for i in category:
#         doc_num = doc_nums[i]
#         prior = doc_num / docs_num
#         for item in voc:
            #count
        # con_prob = x/count