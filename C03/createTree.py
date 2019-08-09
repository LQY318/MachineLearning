import math
import numpy as np

"""
feature既有命名（name），还有编号（index）,以及取值（value）
"""
def shannon_ent(dataset):
    """
    Function to calculate entropy.

    :args dataset: 2Darray including features and label(the last colume)

    :return: float, entropy
    """
    labels = dataset[:, -1]  # 获取标签
    total_num = labels.shape[0]  # 获取样本总数量
    ## 创建字典，存放类别标签及其对应数量
    lb_dict = {}
    for lb in labels:
        if lb not in lb_dict.keys():
            lb_dict[lb] = 0
        lb_dict[lb] += 1
    ## 计算信息熵
    entropy = 0.0
    for k, val in lb_dict.items():
        prob = val / total_num
        entropy -= prob * math.log(prob, 2)

    return entropy

def split_dataset(dataset, feature_idx, value):
    """
    Function to divide the dataset based on a specified value for a particular feature.

    :args dataset: 2D array, Dataset to be divided
    :args feature_idx: int, Index of the particular feature (from zero)
    :args value: int, One of the value about the particular feature

    :return: 2D array, Subset based on a specified value for a particular feature
    """
    subset = None  # 创建空array
    for vec in dataset:  # 得到vector
        if vec[feature_idx] ==  str(value):  # 因为myData中包含str，所以数字也是str,而不是int
            vec_front = vec[:feature_idx]
            vec_rear = vec[feature_idx + 1:]
            new_vec = np.hstack((vec_front, vec_rear))  # 去除指定feature
            if subset is None:
                subset = new_vec[np.newaxis, :]  # 将vector转为2Darray
            else:
                subset = np.vstack((subset, new_vec))

    return subset  # 仍是array，子集中不包含指定的feature，消耗特征

def best_feature(dataset):
    """
    Based on the information gain, choose the best feature as the decision node.

    :args dataset: 2D array, Dataset

    :return: int, Index of the best feature
    """
    n = dataset.shape[1] - 1  # 获取特征数量
    best_info_gain = 0
    base_ent = shannon_ent(dataset)
    for i in range(n):  # 遍历所有特征
        new_ent = 0
        uni_val = set(dataset[:, i])  # 去重，该特征的所有可能取值
        for v in uni_val:  # 遍历该特征所有取值
            subset = split_dataset(dataset, i, v)
            prob = subset.shape[0] / dataset.shape[0]
            new_ent += prob * shannon_ent(subset)
        info_gain = base_ent - new_ent  # 求解每个特征的信息增益
        ## 找到信息增益最大的特征，用于创建一个判断节点
        if info_gain > best_info_gain: 
            best_info_gain = info_gain
            best_f = i

    return best_f

def Majority_vote(labels):
    """
    当所有特征都已经处理完毕，但是标签还是不只一种，此时通过投票法来决定类别。
    什么时候会出现这种情呢？

    :args labels: a list

    :return: int or str..., label
    """
    label_dict = {}
    for lb in labels: 
        if lb not in label_dict.keys(): label_dict[lb] = 0
        label_dict[lb] += 1
    sorted_label = sorted(label_dict.items(), key=lambda x:x[0], reverse=True)  
    # reverse=True降序,返回list,如[('yes', 2), ('no', 1)]

    return sorted_label[0][0]

def create_tree(dataset, feature_Names):
    """
    Function to create tree.

    :args dataset: 2D array, Dataset
    :args feature_Names: list, a list of feature names

    :return: a dict, Tree 
    """
    ## 跳出递归函数的两个条件
    labels = list(dataset[:, -1])
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(dataset[0]) == 1:
        return Majority_vote(labels)
    ## 首先找到当前dataset中最好的特征，作为判断节点
    best_feature_idx = best_feature(dataset)
    best_feature_name = feature_Names[best_feature_idx]

    tree = {best_feature_name:{}}

    sub_feature_Names = feature_Names[:]
    del(sub_feature_Names[best_feature_idx])  # 消耗特征
    # del(feature_Names[best_feature_idx])  # 这个操作不可以！！！

    ## 然后针对最好的特征进行数据集分类，递归调用
    uni_val = set(dataset[:, best_feature_idx])
    for feature_value in uni_val:
        tree[best_feature_name][feature_value] = create_tree(split_dataset(dataset\
            , best_feature_idx, feature_value), sub_feature_Names)

    return tree



if __name__ == "__main__":
    myData = np.array([
        ["1","1","yes"],
        ["1","1","yes"],
        ["1","0","no"],
        ["0","1","no"],
        ["0","1","no"]])
    feature_Names = ["no surfacing", "flippers"]

    # print(shannon_ent(myData))
    # print(split_dataset(myData, 1, 0))
    # print(best_feature(myData))
    # labels = ["yes", "yes", "no"]
    # print(Majority_vote(labels))
    print(create_tree(myData, feature_Names))  
    ## return {'no surfacing': {'1': {'flippers': {'1': 'yes', '0': 'no'}}, '0': 'no'}}
