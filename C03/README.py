createTree.py用于创建树结构
classifyTree.py构建分类器
plotTree.py用于绘制树结构图
storageTree.py用于存储以及加载树结构
Tree.txt则是保存的树结构
main.py是用于测试的主函数，所用数据集在data/目录中


存在的问题：
1.plot_tree.py中的DepthTree函数为什么要进行深度的比较？
2.classifyTree.py中classifier函数：未声明全局变量则返回pred_label时报错未定义pred_label，不知道为什么？
3.plot_tree.py绘制树结构图，但是树结构图有问题，但是我还不知道问题出在哪里
