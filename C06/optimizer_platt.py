import numpy as np
from optimizer_simple import data_fmt, selectJrand, limit_alpha

class optStruct:
    """
    创建一个数据结构，保存重要的值
    """
    def __init__(self, dataset, C, toler, kTup):
        self.features = dataset[:, :-1]
        self.labels = dataset[:, -1]
        self.C = C
        self.toler = toler
        self.m = dataset.shape[0]
        self.alphas = np.zeros(self.m)
        self.b = 0
        self.Cache = np.zeros((self.m, 2))

def calcEk(oS, k):
    """
    计算预测误差

    :args oS: optStruct, data struct
    :args k: int, index of the selected sample

    :return: int, Ek
    """
    fxk = float((oS.alphas*oS.labels).dot(oS.features.dot(oS.features[k, :].T))) + oS.b
    Ek = fxk - float(oS.labels[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    ## 将误差矩阵每一行第一列置１，以此确定出误差不为０的样本
    oS.Cache[i] = [1, Ei]
    ## 获取缓存中Ei不为０的样本对应的alpha列表
    validEcacheList = np.nonzero(oS.Cache[:, 0])[0]
    ## 在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
    if (len(validEcacheList) > 0):
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        ## 否则，就从样本中随机选取alphaj
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS,  k):
    Ek = calcEk(oS, k)
    oS.Cache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labels[i] * Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or \
    ((oS.labels[i] * Ei > oS.toler) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labels[i] != oS.labels[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        eta = 2.0 * oS.features[i, :].dot(oS.features[j, :].T) - \
        oS.features[i, :].dot(oS.features[j, :].T) - \
        oS.features[j, :].dot(oS.features[j, :].T)
        if eta >= 0:
            print ("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labels[j] * (Ej - Ei) / eta
        oS.alphas[j] = limit_alpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labels[j] * oS.labels[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labels[i] * (oS.alphas[i] - alphaIold) * \
        oS.features[i, :].dot(oS.features[i, :].T) - oS.labels[j] * \
        (oS.alphas[j] - alphaJold) * oS.features[i, :].dot(oS.features[j, :])
        b2 = oS.b - Ej - oS.labels[i] * (oS.alphas[i] - alphaIold) * \
        oS.features[i, :].dot(oS.features[j, :].T) - oS.labels[j] * \
        (oS.alphas[j] - oS.alphaJold) * oS.features[j, :].dot(oS.features[j, :].T)
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[j]):
            oS.b = b1
        elif(0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0

def smop(dataset, C, toler, maxIter, kTup=("lin", 0)):
    oS = optStruct(dataset, C, toler, kTup)
    iters = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iters < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: {:d} i: {:d}, pairs changed {:d}".format(iters, i, alphaPairsChanged))
            iters += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas > 0) * (oS.alphas < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: {:d} i: {:d}, pairs changed {:d}".format(iters, i, alphaPairsChanged))
            iters += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: {:d}".format(iters))
    return oS.b, oS.alphas

def weights_svm(alphas, dataset):
    """
    计算权重向量
    """
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    m, n = features.shape
    w = np.zeros(n)
    for i in range(m):
        # import pdb;pdb.set_trace()
        w += alphas[i] * labels[i] * features[i, :]
    return w

def classifier_svm(weights, b, features):
    """
    分类器

    :args weights: vector
    :args b: int
    :args features: vector, one sample

    :return: int, label prediction 
    """
    pred = features.dot(weights) + b
    if pred > 0:
        return 1
    else:
        return -1


if __name__ == "__main__":
    file_name = "./data/testSet.txt"
    dataset = data_fmt(file_name)
    b, alphas = smop(dataset, 0.6, 0.001, 40)
    print(b, alphas)
    weights = weights_svm(alphas, dataset)
    print(weights)