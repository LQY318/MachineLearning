from data_process import num_fmt
from classify import KNN_classifier
import numpy as np
import time

if __name__ == "__main__":

    ## 数据格式化
    trainfile="./data/digits/trainingDigits/"
    testfile="./data/digits/testDigits/"
    trn_feature, trn_label = num_fmt(file_dir=trainfile)
    tst_feature, tst_label = num_fmt(file_dir=testfile)
    trn_Num = trn_feature.shape[0]
    tst_Num = tst_feature.shape[0]

    ## 创建训练集
    res_trainlabel = trn_label.reshape((trn_Num, 1))  # label vector reshape into 2D array 
    train_data = np.hstack((trn_feature, res_trainlabel))  #  stack 
    ## 测试，并记录错误率
    st = time.time()
    err_count = 0
    for i, test_data in enumerate(tst_feature):
        predict_label = KNN_classifier(testdata=test_data, traindata=train_data, k=3)  # 调用KNN分类器
        if predict_label != tst_label[i]: err_count += 1
    err_rate = err_count / tst_Num

    print("Test Result: {:d} error/{:d} total \nError Rate: {:.2f}%".format(err_count, tst_Num, err_rate))
    print("Total time: {:.2f}s.".format(time.time() - st))