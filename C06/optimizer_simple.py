import random
import numpy as np

def data_fmt(file_name):
    """
    data format

    :arg file_name: string, name of data file

    :return: 2Darray including features and label(the last column)
    """
    with open(file_name, "r") as file:
        content = file.readlines()
    ## dataset.shape(m, n),m=number of samples, n=number of features + 1(label)
    dataset = []
    for line in content:
        dataset.append(list(map(float, line.strip().split("\t"))))
    return np.array(dataset)

def selectJrand(i, n):
    """
    Function to select the index of another alpha.

    :args i: int, index of the selected alpha
    :args n: int, number of all the alpha

    :return: int, index of the another alpha
    """
    j = i
    while(j == i):
        j = int(random.uniform(0, n))
    return j

def limit_alpha(alpha, H, L):
    """
    Function to limit the range of some alpha.

    :args alpha: int
    :args H: int, upper limit
    :args L: int, lower limit
 
    :return: int
    """
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L
    return alpha

def SMO_simple(dataset, C, toler, maxIter):
    """
    简易版SMO优化算法

    :args dataset: 2Darray including features and label(the last column)
    :args C: int, slack variable(松弛变量)
    :args toler: int, fault tolerance
    :args maxInter: int, maximun number of iterations

    :returns b: int, bias
    :returns alphas: vector 
    """
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    m, n = features.shape
    ## 初始化
    b = 0
    alphas = np.zeros(m)
    iters = 0
    ## 外循环
    while (iters < maxIter):
        ## 改变的alpha对数
        alphaPairsChanged = 0
        ## 遍历样本
        for i in range(m):
            ## 第一个alpha
            fxi = float((alphas*labels).dot(features.dot(features[i, :].T))) + b  # 计算预测值
            Ei = fxi - float(labels[i])  # 计算误差
            ## 满足KKT条件，则选取第二个alpha进行计算
            if (((labels[i] * Ei < -toler) and (alphas[i] < C)) or ((labels[i] * Ei > toler) and (alphas[i] > 0))):
                j = selectJrand(i, m)  # 随机选取第二个alpha的索引
                fxj = float((alphas*labels).dot((features.dot(features[j, :].T)))) + b  # 计算预测值
                Ej = fxj - float(labels[j])  # 计算误差
                ## 记录两个alpha的原始值，便于后续比较
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                ## 如果两个alpha对应的样本标签不同
                if (labels[i] != labels[j]):
                    # 求出相应的上下边界
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C+alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L == H");continue
                ## 根据公式计算未经剪辑的alphaj
                #--------------------------
                eta = 2 * features[i, :].dot(features[j, :].T) - \
                features[i, :].dot(features[i, :].T) - \
                features[j, :].dot(features[j, :].T)
                # 如果eta>=0，跳出循环
                if eta >= 0:print("eta >= 0");continue
                alphas[j] -= labels[j] * (Ei - Ej) / eta
                alphas[j] = limit_alpha(alphas[j], H, L)
                #--------------------------
                ## 如果改变后的alphaj的值变化不大，跳出本次循环
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                ## 否则计算相应的alphai的值
                alphas[i] += labels[j] * labels[i] * (alphaJold - alphas[j])
                ## 分别计算两个alpha情况下对应的b值
                b1 = b - Ei - labels[i] * (alphas[i] - alphaIold) * \
                features[i, :].dot(features[i, :].T) - labels[j] * \
                (alphas[j] - alphaJold) * features[i, :].dot(features[j, :].T)
                b2 = b - Ej - labels[i] * (alphas[i] - alphaIold) * \
                features[i, :].dot(features[j, :].T) - \
                labels[j] * (alphas[j] - alphaJold) * \
                features[j, :].dot(features[j, :].T)
                ## 如果0<alphai<C,那么b=b1
                if (0 < alphas[i]) and (C > alphas[j]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                ## 如果走到此步，　说明改变了一对alpha的值
                alphaPairsChanged += 1
                print("iter: {:d} i: {:d}, paird changed {:d}".format(iters, i, alphaPairsChanged))
        ## 最后判断是否有改变的alpha对，没有就进行下一次迭代
        if (alphaPairsChanged == 0):
            iters += 1
        ## 否则，　迭代次数置为０,继续循环
        else:
            iters = 0
        print("iteration number: {:d}".format(iters))
    return b, alphas


if __name__ == "__main__":
    file_name = "./data/testSet.txt"
    dataset = data_fmt(file_name)
    b, alphas = SMO_simple(dataset, 0.6, 0.001, 40)
    import pdb;pdb.set_trace()