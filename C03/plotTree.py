import matplotlib.pyplot as plt

## 定义文本框和箭头形式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

## 绘制带箭头的节点
def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords="axes fraction", 
        xytext=centerPt, textcoords="axes fraction",
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotmidtext(cntrPt, parentPt, txtstr):
    """
    绘制两个节点之间的文本

    :args cntrPt: tuple or list(x, y), start point,子节点
    :args parentPt: tuple or list(x, y), end point，父节点
    :args txtstr: string, text information
    """
    ## 计算两个节点连线之间的中间位置
    xmid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    ymid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.ax1.text(xmid, ymid, txtstr)

## 计算叶子节点数
def numLeaf(tree):
    num_leaf = 0
    decision_node = list(tree.keys())[0]  # 此处的key为feature name,也就是decision node
    rest_tree = tree[decision_node]  # rest_tree是一个dict
    ## rest_tree是dict, rest_tree的key为feature value；rest_tree的value为dict则递归，否则为leafnode,+1
    for key in rest_tree.keys():  # 说明不是叶子节点
        if type(rest_tree[key]).__name__ == "dict":
            num_leaf += numLeaf(rest_tree[key])
        else:  # 说明是叶子节点
            num_leaf += 1
    return num_leaf

## 计算树的深度/层数
def depthTree(tree):
    depth = 0
    decision_node = list(tree.keys())[0]
    rest_tree = tree[decision_node]
    for key in rest_tree.keys():
        if type(rest_tree[key]).__name__ == "dict":  # 说明不是叶子节点
            pri_depth = 1 + depthTree(rest_tree[key])
        else:  # 说明是叶子节点
            pri_depth = 1
        ## 为什么要作如下判断？
        if pri_depth > depth: depth = pri_depth
    return depth    

def plotTree(tree, parentPt, nodeText):
    """
    绘制树结构

    :args tree: dict, tree structure
    :args parentPt: tuple or list(x, y), current node
    :args nodeText: middle text
    """
    numleaf = numLeaf(tree) 
    depth = depthTree(tree)
    decision_node = list(tree.keys())[0]
    rest_tree = tree[decision_node]
    ## 计算子节点的坐标
    cntrPt = (plotTree.xoff + (1 + numleaf) / 2 / plotTree.totalw, plotTree.yoff)
    ## 绘制节点之间的文本信息
    plotmidtext(cntrPt, parentPt, nodeText)
    ## 绘制节点
    plotNode(decision_node, cntrPt, parentPt, decisionNode)
    ## 每绘制一次图，将y的坐标减少
    plotTree.yoff = plotTree.yoff - 1 / plotTree.totalw
    for key in rest_tree.keys():
        if type(rest_tree[key]).__name__ == "dict":
            plotTree(rest_tree[key], cntrPt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1 / plotTree.totalw
            plotNode(rest_tree[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotmidtext((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1 / plotTree.totalw

def createPlot(tree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()  # 清空画布
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=True, **axprops)

    plotTree.totalw = float(numLeaf(tree))
    plotTree.totald = float(depthTree(tree))
    plotTree.xoff = -0.6 / plotTree.totalw
    plotTree.yoff = 1.2
    plotTree(tree, (0.5, 1.0), "")
    plt.show()


if __name__ == "__main__":
    tree = {'no surfacing': {'1': {'flippers': {'1': 'yes', '0': 'no'}}, '0': 'no'}}
    createPlot(tree)
    ## 图有问题，但是我还不知道问题出在哪里