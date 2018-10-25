import os
import string
from textblob import TextBlob
from textblob import Word
from nltk.corpus import stopwords


# load data
def load_data(path):
    files = os.listdir(path)
    file_dir = dict()
    for file in files:
        file_list = os.listdir(path + "/" + file)
        # 按原有的文档结构保存
        temp = []
        for file_txt in file_list:
            # print(path + "/" + file + "/" + file_txt)
            doc = open(path + "/" + file + "/" + file_txt, 'r', encoding="ISO-8859-1")
            raw = doc.readlines()
            doc.close()
            # normalization
            raw = str(raw).lower()
            temp.append(raw)
            # if index == 1:
            #     print(temp)
        file_dir[file] = temp
    # print(file_dir)
    return file_dir


file_dir = load_data("../../dataset/20news-18828")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def pre_process(data):
    index = 0
    en_stops = set(stopwords.words('english'))
    for key in data.keys():
        files = data[key]
        for file in files:
            # print(file)
            # tokenization
            senten = TextBlob(str(file).replace("\\n", "").replace("\\\\t", "").replace("\\", "").replace("'", ""))
            file = senten.words
            temp = []
            # 去除停用词和数字
            for word in file:
                if word not in en_stops:
                    if not is_number(word):
                        temp.append(word)
                        file = temp
            # index += 1
            # print(index)
            print(file)
    return data


pre_process(file_dir)

# 计算词频


#使用过滤掉停用词和频率低的单词建立词典

# def dictionary(data):
