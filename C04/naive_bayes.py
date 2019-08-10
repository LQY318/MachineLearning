import numpy as np
from doc_process import create_vocabulary, wordbag2vec, wordset2vec

"""
必须保证conditional_prob,prior_prob,category的顺序是一样的！
"""

def train_nbayes(word2vec, labellist):
    """
    Train the naive bayes model.

    :args word2vec: 2Darray
    :args labellist: list

    :returns conditional_prob: 2Darray, row = length of category, col = length of vocabulary
    :returns prior_prob: vector, length = length of category
    """
    category = list(sorted(set(labellist)))  # 类别:去重、排序、转成列表
    ## 计算每种标签的概率
    dict_label = {}
    for label in category:
        dict_label[label] = labellist.count(label)
    prior_prob = np.array([v/len(labellist) for k, v in dict_label.items()])
    ## 计算某类别下的条件概率
    conditional_prob = np.ones((len(category),word2vec.shape[1]))  # 条件概率array
    words_sum = np.array([2] * len(category))  # 每类文档的字符总数vector
    for i, vec in enumerate(word2vec):
        category_idx = category.index(labellist[i])  # 该文档所属类别的索引
        conditional_prob[category_idx] += vec  # 条件概率二维数组的对应行
        words_sum[category_idx] += np.sum(vec)  # 字符统计向量的对应位置
    words_sum = np.tile(words_sum, (word2vec.shape[1], 1)).T  # 将vector扩展成和conditional_prob一样的shape
    conditional_prob = np.log(conditional_prob / words_sum)

    return conditional_prob, prior_prob

def classifier_nb(word2vec, conditional_prob, prior_prob, category):
    """
    classify using naive bayes.

    :args word2vec: a vector of only a document
    :args conditional_prob: 2Darray, the output of train_nbayes
    :args prior_prob: vector, the output of train_nbayes
    :args category: list including all the classes

    :return: predict label 
    """
    word2vec = np.tile(word2vec, (conditional_prob.shape[0],1))  # 将vector扩展成和conditional_prob一样的shape
    prob = np.sum(word2vec * conditional_prob, axis=1) + np.log(prior_prob)
    prob = prob.tolist()
    idx = prob.index(max(prob))

    return category[idx]


if __name__ == "__main__":
    ## 数据
    traindata = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
              ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
              ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
              ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
              ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
              ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    doclist = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please',\
               'maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid',\
               'my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him',\
               'stop', 'posting', 'stupid', 'worthless', 'garbage',\
               'mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him',\
               'quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    labellist = [0,1,0,1,0,1]
    category = list(sorted(set(labellist)))
    ## 利用训练集生成词汇表
    vocab = create_vocabulary(traindata)
    print("vocabulary:", vocab, "\nLength: {:d}".format(len(vocab)))
    ## 生成训练集词向量
    wordset = []
    for data in traindata:
        wordset.append(wordset2vec(data, vocab))  # 词集向量
    wordset = np.array(wordset)
    # print(wordset)
    # for data in traindata:
    #     wordbag.append(wordbag2vec(data, vocab))  #　词袋向量
    # wordbag = np.array(wordbag)
    # print(wordbag)
    ## 训练naive bayes
    conditional_prob, prior_prob = train_nbayes(wordset, labellist)
    ## 测试一
    testdata = ["love", "my", "dalmation"]
    wordset = wordset2vec(testdata, vocab)
    print("Label: ", classifier_nb(wordset, conditional_prob, prior_prob, category))
    ## 测试二
    testdata = ["stupid", "garbage"]
    wordset = wordset2vec(testdata, vocab)
    print("Label: ", classifier_nb(wordset, conditional_prob, prior_prob, category))