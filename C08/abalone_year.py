import numpy as np
import matplotlib.pyplot as plt
from linear_regression import data_fmt, data_norm, stand_LR, local_LR, ridge_LR, stagewise_LR


if __name__ == "__main__":
    ## 数据加载
    dataset = data_fmt("./data/abalone.txt")
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    m, n = features.shape
    ## 局部加权线性回归————前100个样本用于训练
    # trn_pred1 = local_LR(features[0:100], features[0:100], labels[0:100], k=0.1)  # 训练集的预测 
    # trn_rss1 = np.sum((trn_pred1-labels[0:100])**2)  # 训练集的R方————误差平方和
    # trn_pred2 = local_LR(features[0:100], features[0:100], labels[0:100], k=1.0)  # 训练集的预测 
    # trn_rss2 = np.sum((trn_pred2-labels[0:100])**2)  # 训练集的R方————误差平方和
    # trn_pred3 = local_LR(features[0:100], features[0:100], labels[0:100], k=10)  # 训练集的预测 
    # trn_rss3 = np.sum((trn_pred3-labels[0:100])**2)  # 训练集的R方————误差平方和
    # print("trn_rss1:{:.2f} trn_rss2:{:.2f} trn_rss3:{:.2f}".format(trn_rss1, trn_rss2, trn_rss3))
    ## 局部加权线性回归————前100个样本用于训练，后100个样本用于测试
    # tst_pred1 = local_LR(features[100:200], features[0:100], labels[0:100], k=0.1)  # 测试集的预测结果 
    # tst_rss1 = np.sum((tst_pred1-labels[100:200])**2)  # 测试集的R方————误差平方和
    # tst_pred2 = local_LR(features[100:200], features[0:100], labels[0:100], k=1.0)  # 测试集的预测结果 
    # tst_rss2 = np.sum((tst_pred2-labels[100:200])**2)  # 测试集的R方————误差平方和
    # tst_pred3 = local_LR(features[100:200], features[0:100], labels[0:100], k=10)  # 测试集的预测结果 
    # tst_rss3 = np.sum((tst_pred3-labels[100:200])**2)  # 测试集的R方————误差平方和
    # print("tst_rss1:{:.2f} tst_rss2:{:.2f} tst_rss3:{:.2f}".format(tst_rss1, tst_rss2, tst_rss3))
    ## 结果
    # trn_rss1:57.59 trn_rss2:432.46 trn_rss3:549.26
    # tst_rss1:145662.55 tst_rss2:577.08 tst_rss3:522.02
    ## 数据标准化
    features = data_norm(features)
    labels = labels - np.mean(labels, 0)  # 为什么标签也要标准化，而且只是减去均值？
    ## 岭回归：所有样本用于训练；训练前数据标准化；训练30次，每次训练的lam系数呈指数增长
    # w = []
    # for i in range(30):  # 使用标准化的数据
    #     weights,_ = ridge_LR(features, features, labels ,lam=np.exp(i-10))
    #     w.append(weights)
    # w = np.array(w)  # shape（30，n）
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(w)
    # plt.show()
    ## 向前逐步线性回归，使用标准化的数据
    ws_array = stagewise_LR(features, labels, 1000, 0.005)
    print(ws_array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ws_array)
    plt.show()
    ## 标准线性回归，最小二乘法,使用标准化的数据
    weights, st_pred = stand_LR(features, features, labels)
    print(weights)