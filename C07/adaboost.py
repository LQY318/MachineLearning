import numpy as np

def classifier_stump(features, dimen, thresh_val, thresh_ineq):
    """
    Function to classify based on specific feature and threshold.

    :args features: 2Darray
    :args dimen: int, dimension of the specific feature
    :args thresh_val: float, threshold
    :args thresh_ineq: string, "lt" represents >=, "gt" represents <=

    :return: vector, class results
    """
    m = features.shape[0]
    pred = np.ones(m)
    if thresh_ineq == "lt":  # 小于等于阈值的标签设为-1.0
        pred[features[:, dimen] <= thresh_val] = -1.0
    else:  # 大于等于阈值的标签设为-1.0
        pred[features[:, dimen] >= thresh_val] = -1.0
    return pred  # 分类结果

def create_stump(dataset, D):
    """
    Function to create a best desicion stump based current D vector.
    基于当前的D向量，寻找最好的弱分类器

    :args dataset: 2Darray including features and label(the last colunm)
    :args D: vector, weights for samples

    :returns best_stump: a dict including dimension, threshold and thresh_ineq
    :returns min_err: float, weighted error corresponding to the best stump
    :returns best_pred: vector, prediction labels correponding to the best stump
    """
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    m, n = features.shape
    step_num = 10
    min_err = np.inf
    best_stump = {}
    for dimen in range(n):  # 遍历特征
        ft = features[:, dimen]
        step_size = (max(ft)-min(ft)) / step_num
        thresh_list = np.arange(-1*step_size+min(ft), (step_num+1)*step_size+min(ft), step_size)  # 阈值列表
        for thresh in thresh_list:  # 遍历阈值
            for ineq in ["lt", "gt"]:  # 遍历比较方向
                pred = classifier_stump(features, dimen, thresh, ineq)
                error_vec = np.zeros(m)
                error_vec[pred != labels] = 1.0  # 分类错误的置1,分类正确的置0
                weight_error = D.dot(error_vec)  # 错误分类样本的权重和,错误率为何如此计算？
                # print("dim:{:d}, thresh{:.2f}, ineq:{}, weighted error:{:.2f}".\
                #     format(dimen, thresh, ineq, weight_error))
                if weight_error < min_err:  # 找到最小的权重和，保留对应的单层决策树的信息
                    min_err = weight_error
                    best_stump["dim"] = dimen
                    best_stump["thresh"] = thresh
                    best_stump["ineq"] = ineq
                    best_pred = pred.copy()
    return best_stump, min_err, best_pred   # 返回该D权重向量下，最好的单层决策树的信息

def train_AB(dataset, max_iter):
    """
    Function to create AdaBoost classifier.train=create
    每次迭代都添加一个最好的弱分类器， 直至错误率为0或者超过最大迭代次数

    :args dataset: 2Darray including features and label(the last colunm)
    :args max_iter: int, maximun number of iteration

    :returns AdaBoost: list including the best decision stump every iteration
    :returns acc_pred: vector, confidence at last used to calculate ROC （置信度）
    """
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    m, n = features.shape
    acc_pred = np.zeros(m)  # 加权累加预测值
    D = np.ones(m) / m  # 初始化
    AdaBoost = []  # 存放每次迭代的最好弱分类器
    for i in range(max_iter):
        best_stump, error, weak_pred = create_stump(dataset, D)  # 弱分类器的结果
        print("D:",D)
        ## 计算该若分类器的权重alpha
        alpha = 0.5 * np.log((1-error)/max(error, 1e-16))  # 避免error=0时报错
        best_stump["alpha"] = alpha
        AdaBoost.append(best_stump)
        print("prediction of the best stump", weak_pred)
        ## 更新D向量(训练数据的权值分布)
        class_result = np.ones(m)  # vector
        class_result[weak_pred == labels] = -1  # 记录分类结果，正确置-1,错误置1
        D = D * np.exp(alpha * class_result)  # 另一种写法D = D * np.exp(-1*alpha*labels*weak_pred)
        D = D / D.sum()  # 此处是D的求和有疑问？
        ## 当前预测结果
        acc_pred += alpha * weak_pred  # weak_pred是当前弱分类器的预测结果，acc_pred是当前AdaBoost的分类结果(累积了前面的分类结果)
        print("current prediction:", np.sign(acc_pred))
        ## 当前AdaBoost的错误率
        error_rate = ((np.sign(acc_pred) != labels) * np.ones(m)).sum() / m  # np.sign(acc_pred) != labels得到的是一个布尔型的列表
        ## 还可以写成error_rate = np.zeros(m);error_rate[np.sign(acc_pred)!=labels]=1
        print("current error rate:{:.2f}".format(error_rate))
        ## 如果错误率为0，则退出循环，不再训练
        if error_rate == 0.0:
            break
    return AdaBoost, acc_pred

def classifier_AB(features, AdaBoost):
    """
    Classify based on AdaBoost classifier.

    :args features: 2Darray or vector(only one sample)
    :args AdaBoost: the output of train_AB function

    :return: float, 1.0 or -1.0 predicton label
    """
    m = features.shape[0]
    acc_pred = np.zeros(m)
    for weak_classifier in AdaBoost:
        weak_pred = classifier_stump(features, weak_classifier["dim"], weak_classifier["thresh"], weak_classifier["ineq"])
        acc_pred += weak_classifier["alpha"] * weak_pred
        print("accumulate prediton", np.sign(acc_pred))
    return np.sign(acc_pred)


if __name__ == "__main__":
    features = np.array([[1.0, 2.1],[2.0, 1.1],[1.3, 1.0],[1.0, 1.0],[2.0, 1.0]])
    labels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    dataset = np.hstack((features, labels[:,np.newaxis]))
    ## 测试create_stump函数
    # D = np.ones(5) / 5
    # stump, error, pred = create_stump(dataset, D)
    # print(stump, error, pred)
    ## 测试train_AB函数
    AdaBoost, _ = train_AB(dataset, 9)
    print(AdaBoost)
    ## 测试classifier_AB函数
    test_features = np.array([[5,5],[0,0]])
    pred = classifier_AB(test_features, AdaBoost)
    print(pred)