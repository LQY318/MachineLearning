from data_process import dating_fmt, norm
from classify import KNN_classifier
import numpy as np

if __name__ == "__main__":

    ## 数据格式化
    file_name = "./data/dating/datingTestSet2.txt"
    feature, label = dating_fmt(file_name)
    Num = label.shape[0]  
    ratio = 0.1  # 自行划分训练集与测试集比例
    test_n = int(Num * ratio)

    ## 创建训练集
    norm_trainfeature = norm(ori_data=feature[test_n:, :])  # feature array归一化
    res_trainlabel = label[test_n:].reshape((Num-test_n, 1))  # label vector reshape into 2D array 
    train_data = np.hstack((norm_trainfeature, res_trainlabel))  #  stack
    ## 创建测试集
    norm_testfeature = norm(ori_data=feature[:test_n, :])  # feature array 归一化
    res_testlabel = label[:test_n]  
    ## 测试，并记录错误率
    err_count = 0
    for i, test_data in enumerate(norm_testfeature):
        predict_label = KNN_classifier(testdata=test_data, traindata=train_data, k=3)  # 调用KNN分类器
        if predict_label != res_testlabel[i]: err_count += 1
    err_rate = err_count / test_n

    print("Test Result: {:d} error/{:d} total \nError Rate: {:.2f}%".format(err_count, test_n, err_rate))