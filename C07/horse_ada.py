import numpy as np
from adaboost import train_AB, classifier_AB
import matplotlib.pyplot as plt

def data_fmt(file_name):
    """
    对马儿疾病的原始数据进行格式化
    :args file_name: str
    :return dataset: 2Darray including features and label(the last column)
    """
    with open(file_name, "r") as file:
        content = file.readlines()
    ## dataset.shape(m, n),m=number of samples, n=number of features + 1(label)
    dataset = []
    for line in content:
        dataset.append(list(map(float, line.strip().split("\t"))))

    return np.array(dataset)

def plot_ROC(confidence, ground_truth):
    """
    Plot ROC curve

    :args confidence: vector
    :args ground_truth: vector
    """
    cur = (1.0, 1.0)  # 光标
    ySum = 0.0 # 用于计算AUC
    numPosClass = ground_truth[ground_truth==1.0].shape[0]  # 筛选出正例，并统计数目
    ## 为什么x、y的步长这么计算？
    yStep = 1 / numPosClass
    xStep = 1 / (ground_truth.shape[0]-numPosClass)
    sorted_idx = confidence.argsort()  # 对置信度进行小->大的排序，返回对应的索引
    ## 画图
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_idx:
        if ground_truth[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ## 画线cur->(cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c="b")
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1], [0,1], "b--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve for AdaBoost horse colic detection system")
    ax.axis([0,1,0,1])  # axis([xmin, xmax, ymin, ymax])
    print("Area under the curve: {:.2f}".format(ySum*xStep))
    plt.show()



if __name__ == "__main__":
    ## trian
    trn_dataset = data_fmt("./data/horseColicTraining2.txt")
    trn_features = trn_dataset[:, :-1]
    trn_labels = trn_dataset[:, -1]
    AdaBoost, trn_pred = train_AB(trn_dataset, 50)
    print("AdaBoost:",AdaBoost)
    ## plot ROC curve
    plot_ROC(trn_pred, trn_labels)
    ## test
    tst_dataset = data_fmt("./data/horseColicTest2.txt")
    tst_features = tst_dataset[:, :-1]
    tst_labels = tst_dataset[:, -1]
    m = tst_labels.shape[0]  # 测试样本数
    pred = classifier_AB(tst_features, AdaBoost)
    err = np.zeros(m)  # 统计错误率
    err[pred != tst_labels] = 1.0
    error_rate = err.sum() / m
    print("Error rate of testing: {:.2f}".format(error_rate)) 
