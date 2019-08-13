import numpy as np
from logistic_regression import SGD, classifier_LR

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

if __name__ == "__main__":
    ## 获取训练集
    trn_file_name = "./data/horseColicTraining.txt"
    trn_dataset = data_fmt(trn_file_name)
    trn_features = trn_dataset[:, :-1]
    trn_labels = trn_dataset[:, -1]
    ## 训练算法，获得权重
    weights = SGD(trn_features, trn_labels, 500)
    ## 获取验证集
    val_file_name = "./data/horseColicTest.txt"
    val_dataset = data_fmt(val_file_name)
    val_features = val_dataset[:, :-1]
    val_labels = val_dataset[:, -1]
    ## 使用分类器进行验证，计算错误率
    error_num = 0
    for val_ft, val_lb in zip(val_features, val_labels):
        pred_lb = classifier_LR(val_ft, weights)
        if pred_lb != val_lb:
            error_num += 1
    error_rate = 100 * (error_num / val_dataset.shape[0])
    print("Error rate: {:.2f}".format(error_rate))
