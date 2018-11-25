# Data Mining project

# 1. KNN文档分类

1）预处理文本数据集，并且得到每个文本的VSM表示。



2）实现KNN分类器，测试其在20Newsgroups上的效果。训练集:测试集 = 8:2

最好结果：
acc = 0.86 (cosine distance，K <= 30时 acc > 0.82）
acc = 0.70（euclidean distance, K <= 4时 acc > 0.5)

==================================================

# 2. Naive Bayes分类
利用实验1得到的词频，词典等信息完成文档分类。
测试其在20Newsgroups上的效果。训练集:测试集 = 8:2
