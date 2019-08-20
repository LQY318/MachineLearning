import numpy as np
import matplotlib.pyplot as plt

def data_fmt(file_name):
    """
    对原始数据进行格式化

    :args file_name: string

    :return dataset: 2Darray including features and label(the last column)
    """
    with open(file_name, "r") as file:
        content = file.readlines()
    ## dataset.shape(m, n),m=number of samples, n=number of features + 1(label)
    dataset = []
    for line in content:
        dataset.append(list(map(float, line.strip().split("\t"))))

    return np.array(dataset)

def data_norm(ori_data):
    """
    数据标准化normalization

    :args ori_data: 2Darray or vaector

    :return norm_data: 2Darray, normalized data
    """
    mean = np.mean(ori_data, 0)
    var = np.var(ori_data, 0)
    norm_data = (ori_data - mean) / var
    return norm_data

def stand_LR(tst_features, trn_features, trn_labels):
    """
    Function to calculate weights(regression coeffients) and predict based on local linear regression.
    所有样本共用weights vector

    :args tst_features: 2Darray or vector,需要预测的样本点，既可以是训练集的特征，也可以是测试集的特征
    :args features: 2Darray，训练集的特征
    :args labels: vector，训练集的标签

    :return weights： vector, 权重
    :return tst_pred: vector,预测的标签
    """
    xTx = (trn_features.T).dot(trn_features)
    ## 根据行列式是否为０判断是否可以转置
    if np.linalg.det(xTx) == 0.0:
        print("This 2D array is singualr, cannot do inverse.")
        return
    weights = np.linalg.inv(xTx).dot(trn_features.T).dot(trn_labels)
    tst_pred = [tst_x.dot(weights) for tst_x in tst_features]
    return weights, np.array(tst_pred)

def local_LR(tst_features, trn_features, trn_labels, k):
    """
    Function to calculate weights(regression coeffients) and predict based on local linear regression.
    每个样本都有各自的weights vector

    :args tst_features: 2Darray or vector,需要预测的样本点，既可以是训练集的特征，也可以是测试集的特征
    :args features: 2Darray，训练集的特征
    :args labels: vector，训练集的标签
    :args k: float, 平滑系数,越小越平滑

    :return tst_pred: vector,预测的标签
    """
    m = trn_features.shape[0]  # m个训练样本
    tst_pred = []
    for tst_x in tst_features:
        w = np.eye(m)
        for j, trn_x in enumerate(trn_features):
            distance = np.sum((tst_x - trn_x) ** 2)  # 为什么原文此处不开根号？
            w[j, j] = np.exp(distance / (-2 * k**2))
        xTx = (trn_features.T).dot(w).dot(trn_features)
        if np.linalg.det(xTx) == 0.0:
            print("This 2D array is singualr, cannot do inverse.")
            return
        weights = np.linalg.inv(xTx).dot(trn_features.T).dot(w).dot(trn_labels)
        tst_pred.append(tst_x.dot(weights))
    return np.array(tst_pred)

def ridge_LR(tst_features, trn_features, trn_labels, lam=0.2):
    """
    Function to calculate weights(regression coeffients) and predict based on ridge linear regression.
    所有样本共用weights vector

    :args tst_features: 2Darray or vector,需要预测的样本点，既可以是训练集的特征，也可以是测试集的特征
    :args features: 2Darray，训练集的特征
    :args labels: vector，训练集的标签
    :args lam: float, 单位矩阵的系数

    :return weights： vector, 权重
    :return tst_pred: vector,预测的标签
    """
    m, n = trn_features.shape
    xTx = (trn_features.T).dot(trn_features)
    ridge = xTx + np.eye(n) * lam
    ## 根据行列式是否为０判断是否可以转置
    if np.linalg.det(ridge) == 0.0:  # 当lam=0.0时，仍需要判断
        print("This 2D array is singualr, cannot do inverse.")
        return
    weights = np.linalg.inv(ridge).dot(trn_features.T).dot(trn_labels)
    tst_pred = [tst_x.dot(weights) for tst_x in tst_features]
    return weights, np.array(tst_pred)

def stagewise_LR(trn_features, trn_labels, numIt=100, eps=0.01):
    """
    Function to calculate weights(regression coeffients) and predict based on local linear regression.
    所有样本共用weights vector;进行numIt次迭代训练，每次迭代都将逐个对特征加减（sign）一个很小的值（eps）

    :args trn_features: 2Darray，训练集的特征
    :args trn_labels: vector，训练集的标签
    :args numIt: int, 迭代次数
    :args eps: float, 每次迭代需要调整的步长

    :return ws_array: 2Darray, shape(numIt, n)
    """
    m, n = trn_features.shape
    ws = np.zeros(n)
    ws_array = np.zeros((numIt, n))
    for i in range(numIt):
        print(ws)
        lowest_rss = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_c = ws.copy()  # 因为每次都是对ws进行加减一个很小的值，所以要先copy，对copy的权重进行操作
                ws_c[j] += sign * eps
                pred = trn_features.dot(ws_c)
                rss = np.sum((pred - trn_labels) ** 2)
                if rss < lowest_rss:
                    lowest_rss = rss
                    ws_max = ws_c  # 若当前的权重组合能得到最小的rss，则将ws_c保存起来
        ws = ws_max.copy()  # 每次迭代都只对一个特征进行了增加或减少eps的操作
        ws_array[i] = ws
    return ws_array


if __name__ == "__main__":
    ## 加载数据
    dataset = data_fmt("./data/ex0.txt")
    features = dataset[:, :-1]  # features的第一列是1.0，常数项
    labels = dataset[:, -1]
    m, n = features.shape
    ## 测试stand_LR
    # weights, st_pred = stand_LR(features, features, labels)
    # print(st_pred[0])
    ## 画图
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(features[:, 1], labels)  # 画散点图（取features的第二列）
    # X = features.copy()  # 拟合曲线要按顺序来画，所以要进行排序
    # X.sort(0)  # 排序
    # pred = X.dot(weights)
    # ax.plot(X[:, -1], pred)
    # plt.show()
    ## 相关系数
    # cor = np.corrcoef(pred, labels)
    # print(cor)
    ## 误差平方和
    # print(np.sum((pred - labels) ** 2))  
    ## 测试local_LR
    pred = local_LR(features, features, labels, 0.001)  # 预测训练样本的标签
    print(pred[0])
