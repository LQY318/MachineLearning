import numpy as np

def KNN_classifier(testdata, traindata, k):
    """
    Function to classify.

    :args testdata: a vector, feature of only a test sample
    :args traindata: a 2D array including feature and label(the last colume)
    :args k: the nearest k samples
 
    :return: a int, label of the test sample
    """
    Num = traindata.shape[0]
    np.tile(testdata, (Num, 1))
    # element-wise 计算距离，返回vector
    distance = (((testdata - traindata[:, :-1]) ** 2).sum(axis=1)) ** 0.5
    # 对距离vector排序，并返回对应索引
    sortedDistance_index = np.argsort(distance)
    # 得到距离最近的ｋ个样本的标签，出现频率最高的标签作为结果
    labels = [traindata[idx][-1] for idx in sortedDistance_index[:k]]
    predict_label = int(max(labels, key=labels.count))  # 标签一般为整数int

    return predict_label


if __name__ == "__main__":

    train_feature = np.array([[1.0,1.1], [1.0,1.0], [1.1,1.2], [0,0], [0,0.1], [0.2,0.2]])
    train_label = np.array([1, 1, 1, 2, 2, 2]).reshape((6,1))  # 使得feature与label有一样的维度，才可以hstack
    train_data = np.hstack((train_feature, train_label))  # 创建feature与label在一起的数据集
    test_data = np.array([1.2, 1.2])

    predict_label = KNN_classifier(testdata=test_data, traindata=train_data, k=2)
    print("Predict Label: ", predict_label)