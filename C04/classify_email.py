import re
import random
import numpy as np
import codecs
from doc_process import create_vocabulary, wordbag2vec, wordset2vec, DelHighFreq
from naive_bayes import train_nbayes, classifier_nb

def dataparse(filename):
    """
    解析文本：分割、去标点、转小写
    """
    try:
        with open(filename, "r") as file:
            content = file.read()
    # ./email/ham/6.txt
    # ./email/spam/17.txt
    # ./email/ham/23.txt
    # 读取这三个文件的内容会报编码错误，不知道为什么？
    except:
        print(filename)
        return None
    tokens = re.split("\W*", content)
    return [t.lower() for t in tokens if len(t) > 2]

def getTopWords(vocabulary, conditional_prob):
    """
    获得每类文档出现概率最高的词汇。

    :args vocabulary: list, vocabulary
    :args conditional_prob: 2Darray, H=number of classes, W=number of vocabulary

    :return: 2D list
    """
    topWord = []
    for i, prob_list in enumerate(conditional_prob):
        idx_list = np.squeeze(np.argwhere(prob_list > -4.4))  # 2Darray  -> vector -4.4这个值视情况而定
        topWord.append([vocabulary[idx] for idx in idx_list])  # 该类别文本的高频词
    return topWord


if __name__ == "__main__":
    docdata = []
    fulldata = []
    labellist = []
    ## 导入并解析文件内容
    for i in range(1,26):
        ## 读取spam类型的邮件内容
        doc_per = dataparse("./email/spam/%d.txt" % i)
        if doc_per is None:  # 跳过编码错误的文件
            continue
        docdata.append(doc_per)
        fulldata.extend(doc_per)
        labellist.append(1)
        ## 读取ham类型的邮件内容
        doc_per = dataparse("./email/ham/%d.txt" % i)
        if doc_per is None:  # 跳过编码错误的文件
            continue
        docdata.append(doc_per)
        fulldata.extend(doc_per)
        labellist.append(0)
    doc_num = len(docdata)  # 文档的数量
    category = list(sorted(set(labellist)))
    ## 生成词汇表
    vocab = create_vocabulary(docdata)
    vocab = DelHighFreq(vocab, fulldata, top_num=30)
    ## 交叉验证，随机划分训练集和验证集
    trainingSet = list(range(doc_num))  # trainingSet用于存放训练文档的索引
    testSet = []  # testSet用于存放测试文档的索引
    for i in range(int(doc_num/5)):  # 1/5作为测试文档
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    ## 创建训练集，包括文档词向量和文档标签
    trainData = []
    trainLabels = []
    for docIndex in trainingSet:   
        trainData.append(wordset2vec(docdata[docIndex], vocab))
        trainLabels.append(labellist[docIndex])
    ## 训练naive bayes
    conditional_prob, prior_prob = train_nbayes(np.array(trainData), trainLabels)
    ## 测试
    error_cnt = 0
    for docIndex in testSet:  # 每次对一个文档进行测试
        word_vec = wordset2vec(docdata[docIndex], vocab)
        pred_label = classifier_nb(word_vec, conditional_prob, prior_prob, category)
        if pred_label != labellist[docIndex]:
            error_cnt += 1
    print("Error rate: {:.2f}%".format(error_cnt / len(testSet) * 100))
    ## 获取高频词汇
    top_word = getTopWords(vocab, conditional_prob)
    for words, categ in zip(top_word, category):
        print("Category : {}\nTop words:{}".format(categ, words))