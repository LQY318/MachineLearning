def classifier(tree, feature_Names, testvec):
    """
    classifier for testdata

    :args tree: dict, tree structure
    :args feature_Names: list, a list of feature names
    :args testvec: a list

    :return: a label 
    """
    decision_node = list(tree.keys())[0]
    rest_tree = tree[decision_node]
    feature_idx = feature_Names.index(decision_node)

    for key in rest_tree.keys():
        # key是该特征的取值，可能是str数据类型
        if key == testvec[feature_idx]:
            if type(rest_tree[key]).__name__ == "dict":  # 是判断节点，递归调用
                classifier(rest_tree[key], feature_Names, testvec)
            else:  # 是叶子节点，作为预测标签
                global pred_label  # 未声明全局变量则返回pred_label时报错未定义pred_label，不知道为什么？
                pred_label = rest_tree[key]
                # print(pred_label)
                
    return pred_label  


if __name__ == "__main__":
    tree = {'no surfacing': {'1': {'flippers': {'1': 'yes', '0': 'no'}}, '0': 'no'}}
    feature_Names = ["no surfacing", "flippers"]
    testvec = ["1", "0"]
    print(classifier(tree, feature_Names, testvec))