import os
import math
from textblob import TextBlob
from textblob import Word
from nltk.corpus import stopwords

path = "../../dataset/20news-18828"


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
def pre_process(data):
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


pre_data = pre_process(file_dir)
# pre_process(file_dir)


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


# 计算全局文档词频，返回一个字典
def cal_global(dict_tf):
    dict_fold = dict()
    for key in dict_tf:
        files = dict_tf[key]
        for file in files:
            for key in file['file'].keys():
                if key not in dict_fold.keys():
                    frequency = [0, 0]
                    frequency[0] += 1
                    frequency[1] += file['file'][key]
                    dict_fold[key] = frequency
                else:
                    dict_fold[key][0] += 1
                    dict_fold[key][1] += file['file'][key]

    # print(dict_fold)
    # print(len(dict_fold))
    return dict_fold


# 使用过滤掉停用词和频率低的单词建立词典
def filter_dict(directory):
    final_dict = directory
    result = open("../../data/dict.txt", 'w', encoding='utf-8')
    all_str = ''
    for key in directory.copy():
        if directory[key][1] < 15:
            final_dict.pop(key)
        else:
            all_str += str(key) + '\n'
    # print(final_dict)
    # print(len(final_dict))
    result.write(all_str)
    result.close()
    return final_dict


current = cal_current(pre_data)
# cal_current(pre_data)
globaldict = cal_global(current)
final_dict = filter_dict(globaldict)


# 计算tf-idf,得到每个文档的VSM表示
def cal_tfidf(final_dict, current):
    for key in current:
        files = current[key]
        file_list = os.listdir(path + "/" + key)
        # 计算tf
        os.mkdir("../../data/" + key)
        for index, file in enumerate(files):
            result = open("../../data/" + key+'/' + file_list[index], 'w')
            length = file['len']
            now_dict = file['file']
            VSM = ''
            for word in final_dict.keys():
                if word in now_dict.keys():
                    tf = now_dict[word]/length
                    idf = math.log(18828/(final_dict[word][0]+1))
                    tf_idf = tf * idf
                    # print(tf_idf)
                else:
                    tf_idf = 0
                VSM += str(tf_idf) + '\n'
            result.write(VSM)
            result.close()


# cal_tfidf(final_dict, current)


