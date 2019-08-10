import numpy as np

def create_vocabulary(docmat):
    """
    Function to create the vocabulary for the document dataset.

    :args docmat: matrix, store one document per line

    :return: list including all the words, used as a list of feature names
    """
    ## 遍历所有文档内容，生成列表
    # vocabulary = []
    # for doc in docmat:
    #     for word in doc:
    #         if word not in vocabulary:
    #             vocabulary.append(word)
    ## 求解集合set的并集
    vocabulary = set([])
    for doc in docmat:
        vocabulary = vocabulary | set(doc)

    ## 按照词汇的首字母进行排序，否则每次生成的词汇表顺序不一样
    return sorted(list(vocabulary))  

def wordset2vec(docdata, vocabulary):
    """
    Function to create the set-of-words vector.

    :args docdata: list
    :args vocabulary: list

    :return: 2Darray, the same rows as the docdata, one-hot
    """  
    vec = [0] * len(vocabulary) 
    for word in docdata:
        if word not in vocabulary:  # 测试集会出现某单词不在词汇表中的情况，此时跳过该单词
            continue
        vec[vocabulary.index(word)] = 1  # 存在，则相应位置置１
    return np.array(vec) 

def wordbag2vec(docdata, vocabulary):
    """
    Function to create the bag-of-words vector.

    :args docdata: list
    :args vocabulary: list

    :return: 2Darray, height is the number of the document, width is the length of the vocabulary.
             Count the number of times a word appears.
    """  
    vec = [0] * len(vocabulary)
    for word in docdata:
        if word not in vocabulary:  # 测试集会出现某单词不在词汇表中的情况，此时跳过该单词
            continue
        vec[vocabulary.index(word)] += 1  # 统计单词出现的个数
    return np.array(vec)  

def DelHighFreq(vocabulary, doclist, top_num):
    """
    Function to delete high frequency words.

    :args vocabulary: list
    :args doclist: 1D list, the same content as docmat(matrix)

    :return: list, vocabulary without high frequency words 
    """
    ## 创建字典，进行统计并排序
    dict_vocab = {}
    for token in vocabulary:
        dict_vocab[token] = doclist.count(token)  # 对list的指定元素进行统计
    sorted_freq = sorted(dict_vocab.items(), key=lambda x: x[1], reverse=True)  # 对字典进行排序
    # print(sorted_freq)
    """
    [('dog', 3), ('him', 3), ('my', 3), ('stupid', 3),
    ('stop', 2), ('to', 2), ('worthless', 2),
    ('I', 1), ('ate', 1), ('buying', 1), ('cute', 1)
    ('dalmation', 1), ('flea', 1), ('food', 1),
    ('garbage', 1), ('help', 1), ('how', 1), ('has', 1),
    ('is', 1), ('love', 1), ('licks', 1), ('mr', 1),
    ('maybe', 1), ('not', 1), ('problems', 1), ('posting', 1),
    ('park', 1), ('please', 1), ('quit', 1), ('steak', 1),
    ('so', 1), ('take', 1)]
    """
    ## 删除top_num的高频词汇
    top_vocab = sorted_freq[:top_num]
    for word in top_vocab:
        if word[0] in vocabulary: vocabulary.remove(word[0])

    return vocabulary


if __name__ == "__main__":
    docmat = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
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

    vocab = create_vocabulary(docmat)
    # print("vocabulary:", vocab, "\nLength: {:d}".format(len(vocab)))

    # wordset = wordset2vec(docmat, vocab) 
    # print(wordset)

    # wordbag = wordbag2vec(docmat, vocab)
    # print(wordbag)

    rest_vocab = DelHighFreq(vocab, doclist, 3)
    print(rest_vocab, "\nLength:", len(rest_vocab))