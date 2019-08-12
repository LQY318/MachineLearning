import numpy as np
import random

def sigmoid(inp):
    """
    计算sigmoid的值

    :args inp: int or vector

    :return: int or vector
    """
    p = 1 / (1 + np.exp(-inp))
    return p

def SGD(features, labels, iter_num):
    """
    随机选取样本进行梯度更新

    :args features: 2D array,shape(m, n),m=number of samples,n=number of features
    :args labels: vector, shape(m,)
    :args iter_num: int, number of iteration

    :return: weights vector, shape(n,)
    """
    ## 初始设置
    m, n = features.shape  # 获取样本数以及特征数
    weights = np.ones(n)  # 权重初始化为1.0
    ## 迭代学习
    for i in range(iter_num):
        data_idx = list(np.arange(m))  # 存放数据的索引，每选中一个样本，就删除对应索引
        for j in range(m):
            alpha = 4 / (1 + j + i) + 0.01  # 设置学习率
            idx = int(random.uniform(0, len(data_idx)))  # 随机获取索引列表的索引
            pred = sigmoid(np.sum(features[data_idx[idx]] * weights))  # 计算预测值
            error = labels[data_idx[idx]] - pred  # 计算误差
            weights = weights + alpha * error * features[data_idx[idx]]  # 进行梯度更新
            del(data_idx[idx])  # 删除对应的样本索引

    return weights

def classifier_LR(datavec, weights):
    """
    分类器

    :args datavec: vector(n,), n=number of features
    :args weight: vector(n,)

    :return: int, prediction 
    """
    pred = sigmoid(np.sum(datavec * weights))
    if pred > 0.5:
        return 1.0
    else:
        return 0.0
