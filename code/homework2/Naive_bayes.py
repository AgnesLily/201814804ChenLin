import os
import math
from textblob import TextBlob
from textblob import Word
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
path = "../../dataset/20news-18828"
file_path = "../../data2"


# 加载数据
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
            row = doc.readlines()
            doc.close()
            # normalization
            row = str(row).lower()
            temp.append(row)
            # if index == 1:
            #     print(temp)
        file_dir[file] = temp
    # print(file_dir)
    return file_dir


file_dir = load_data(path)


# 读入词典
def load_dict():
    doc = open('../../data/dict.txt', 'r', encoding="ISO-8859-1")
    dictionary = set()
    line = doc.readline()
    while line:
        dictionary.add(line.replace("\n", ""))
        line = doc.readline()
    # print(dictionary)
    return dictionary


dictionary = load_dict()


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


# 对文档中的词进行预处理
def pre_process(data, dictionary):
    # print(dictionary)
    # index = 0
    en_stops = set(stopwords.words('english'))
    fold_dict = dict()
    for key in data.keys():
        files = data[key]
        fold_dict[key] = []
        for file in files:
            # print(file)
            # tokenization
            senten = TextBlob(str(file).replace("\\n", "").replace("\\t", "").replace("\\", "").replace("'", ""))
            file = senten.words
            temp = []
            # 去除停用词，数字，以及stemming
            for word in file:
                if word not in en_stops:
                    if word in dictionary:
                        if not is_number(word):
                            word = Word(word).lemmatize()
                            word = Word(word).lemmatize('v')
                            temp.append(''+str(word))
            file = temp
            fold_dict[key].append(file)
            # index += 1
            # print(index)
            # print(file)
    return fold_dict


pre_data = pre_process(file_dir, dictionary)


# 计算当前文档词频，返回一个字典
def cal_current(data):
    dict_fold = dict()
    for key in data.keys():
        files = data[key]
        frequency = []
        for file in files:
            dict_file = dict()
            dict_cur = dict()
            fiel_len = len(file)
            dict_file['len'] = fiel_len
            for word in file:
                # print(word)
                if word not in dict_cur.keys():
                    dict_cur[word] = 0
                    for other in file:
                        if other == word:
                            dict_cur[word] += 1
            dict_file['file'] = dict_cur
            frequency.append(dict_file)
        dict_fold[key] = frequency
        # for keys, values in dict_fold.items():
        #     print(keys)
        #     print(values)
        # print(dict_fold)
    return dict_fold


sta_word = dict()


# 计算每个类的词频，并存成文件
def cal_global(dict_tf):
    global sta_word
    for key in dict_tf:
        temp = [0, 0]
        dict_fold = dict()
        files = dict_tf[key]
        # print(key)
        result = open("../../data2/" + key + '.txt', 'w', encoding='utf-8')
        count_word = 0
        all_words = 0
        all_str = ''
        for file in files:
            # 每个类中的每个文件
            for item in file['file'].keys():  # 遍历每个文档的词频表
                if item not in dict_fold.keys():
                    count_word += 1
                    dict_fold[item] = file['file'][item]
                    all_words += file['file'][item]
                else:
                    dict_fold[item] += file['file'][item]
                    all_words += file['file'][item]
        for i in dict_fold.keys():
            all_str += str(i) + ' ' + str(dict_fold[i]) + '\n'
        # print(dict_fold)
        temp[0] = count_word
        temp[1] = all_words
        sta_word[key] = temp
        result.write(all_str)
        result.close()
    print(len(sta_word))


current = cal_current(pre_data)
cal_global(current)


def process_data(pre_data):
    train_x = []
    train_y = []
    # index = 0
    label_index = 0
    labels = os.listdir(path)
    for key in pre_data.keys():
        # labels = os.listdir(path)
        file_list = pre_data[key]
        for file in file_list:
            temp = []
            for word in file:
                temp.append(word)
            train_x.append(temp)
            train_y.append(labels[label_index])
            # index += 1
        label_index += 1
    return train_x, train_y


# 划分数据集
train_samples, train_labels = process_data(pre_data)
train_x, test_x, train_y, test_y = train_test_split(train_samples, train_labels, test_size=0.2, random_state=1)


def cal_prior(path):
    category = dict()
    cates = os.listdir(path)
    for i in cates:
        file_list = os.listdir(path + "/" + i)
        num = len(file_list)
        category[i] = math.log(num / 18828)
    return category


dict_category = cal_prior(path)


def cal_prob(path):
    # print(sta_word)
    prob = dict()
    files = os.listdir(path)
    classes = []
    # print(files)
    for file in files:
        item = file.replace('.txt', '')
        classes.append(item)
        prob[item] = dict()
        doc = open(path + "/" + file, 'r', encoding="ISO-8859-1")
        line = doc.readline()
        while line:
            row_list = line.split(' ')
            temp = (float(row_list[1].replace('\n', '')) + 1) / (sta_word[item][0] + sta_word[item][1])
            line = doc.readline()
            prob[item][row_list[0]] = math.log(temp)
    # print(prob)
    return prob, classes


dict_prob, classes = cal_prob(file_path)


def naive_bayes(test_x, test_y, dict_prob, dict_category, classes):
    # print(test_x)
    # print(test_y)
    test_result = []
    for file in test_x:
        result = []
        for cate in classes:
            # print(cate)
            prob = 0
            for term in file:
                if term in dict_prob[cate].keys():
                    prob += dict_category[cate] + dict_prob[cate][term]
            result.append(prob)
        # print(result)
        index = result.index(max(result))
        test_result.append(classes[index])
    count = 0
    for i in range(len(test_result)):
        if test_result[i] == test_y[i]:
            count += 1
        # print(count)
    acc = count / len(test_result)
    print("The acc of naive bayes is:" + str(acc))


naive_bayes(test_x, test_y, dict_prob, dict_category, classes)
