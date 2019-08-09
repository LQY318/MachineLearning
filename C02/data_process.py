import numpy as np
import os
import matplotlib.pyplot as plt

def dating_fmt(file_name):
    """
    Function to parse the data about dating.

    :arg file_name: File name of the original dataset

    :returns feature_m: a 2D matrix about feature 
    :returns label_v: a vector label
    """
    feature = []
    label = []
    ## 读取文件内容
    with open(file_name, "r") as file:
        content = file.readlines()
    for i in range(len(content)):
        ## 处理文件内容
        feature.append(list(map(float, (content[i].strip().split("\t")[:3]))))  # 将list的str元素转为float元素
        label.append(int(content[i].strip().split("\t")[-1]))  # 将str转为int
    ## 转为array
    feature_m = np.array(feature)  # N*3 2Darray
    label_v = np.array(label)  # N vector

    return feature_m, label_v

def num_fmt(file_dir):
    """
    Function to parse the data about pictures of handwritten number.

    :arg file_dir: File diretory of the original dataset

    :returns feature_m: a 2D matrix about images 
    :returns label_v: a vector label
    """
    feature = []
    label = []

    ## 获取文件夹目录列表
    dir_list = os.listdir(file_dir)
    for i in range(len(dir_list)):
        ## 读取文件内容
        with open(file_dir + dir_list[i]) as img_file:
            img = img_file.read()
        ## 处理文件内容
        feature.append([int(j) for j in img if j != "\n"])  # 去除换行符，数字转为int
        label.append(int(dir_list[i].split("_")[0]))
    ## 转为array
    feature_m = np.array(feature)  # N*(32*32) 2Darray
    label_v = np.array(label)  # N vector

    return feature_m, label_v

def analyze(feature, label):

    # 创建画布figure
    fig = plt.figure()
    # 创建子图axes
    ax = fig.add_subplot(111)
    # 绘制散点图，并指定每一类的点的大小与颜色
    ax.scatter(feature[:, 1], feature[:, 2], s=10.0 * label, c=label)
    # 每个轴的上下限
    ax.axis([-2, 25, -0.2, 2.0])
    # x y轴的标注
    plt.xlabel("Percentage of Time Spent Playing Video Games")
    plt.ylabel("Liter of Ice Cream Consumed Per Week")
    # 显示
    plt.show()

def norm(ori_data):
    """
    Function to normalize the dataset.

    :arg ori_data: original data

    :return: nomalized data
    """
    # 获取样本数量
    Num = ori_data.shape[0]
    #　获取每一列最值，并生成和原始数据size一样的array————并行运算
    max_v = np.max(ori_data, axis=0)
    np.tile(max_v, (Num, 1))
    min_v = np.min(ori_data, axis=0)
    np.tile(min_v, (Num, 1))
    ranges = max_v - min_v
    # 归一化到0-1
    norm_data = (ori_data - min_v) / ranges 

    return norm_data

if __name__ == "__main__":
    ## 逐个函数进行测试
    feature, label = dating_fmt(file_name="./data/dating/datingTestSet2.txt")
    # feature, label = num_fmt(file_dir="./data/digits/trainingDigits/")
    # feature, label = num_fmt(file_dir="./data/digits/testDigits/")
    # analyze(feature=feature, label=label)
    norm_feature = norm(ori_data=feature)
    import pdb;pdb.set_trace()