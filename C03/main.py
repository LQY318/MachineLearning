from createTree import *
from plotTree import *
from classifyTree import *
from storageTree import *
import numpy as np

if __name__ == "__main__":
    ## 获取数据
    DataFile = "./data/lenses.txt"
    with open(DataFile, "r") as data_file:
        content = data_file.readlines()
    dataset = np.array([line.strip().split("\t") for line in content])
    feature_Names = ["age", "prescript", "astigmatic", "tearRate"]
    ## 创建树结构
    tree = create_tree(dataset, feature_Names)
    print(tree)
    ## 保存树结构
    TreeFile = "Tree.txt"
    store_tree(tree, TreeFile)
    ## 加载树结构
    tree = grab_tree(TreeFile)
    print(tree)
    ## 绘制树结构
    createPlot(tree)
    ## 测试树结构
    print(classifier(tree, feature_Names, ["young", "hyper", "yes", "reduced"]))