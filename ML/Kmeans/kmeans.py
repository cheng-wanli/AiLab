# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:28:33 2020

@author: Administrator
"""

'''
Kmeans聚类算法:
    1、随机选取K个聚类中。选取的点可以不是数据点，但一定要在数据的范围内。
    2、计算每个数据点到质心的距离，将距离近的点分配给该类。
    3、重新计算每一类的质心（类中所有点的均值），重复2步骤，直到数据点的类别不再发生变化。
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadDataSet(path):
    dataSet = pd.read_csv(path, sep='\t', header=None)
    dataSet = dataSet.values
    return dataSet

def creatInitialCent(dataSet, k):
    #数据集的列数
    n = dataSet.shape[1]
    initialCent = np.zeros((k, n))
    for i in range(n):
        colMin = dataSet[:, i].min()
        #计算每一维的范围
        scope = dataSet[:, i].max() - colMin
        #按列对初始质心进行赋值
        initialCent[:, i] = (colMin + scope * np.random.rand(k, 1))[:, 0]
    return initialCent

def distance(dataA, dataB):
    # dataA, dataB为数组， 计算两者之间的距离
    dis = np.sqrt(sum(np.power((dataA - dataB), 2)))
    return dis

def kMmeans(dataSet, k):
    # 初始化质心
    centre = creatInitialCent(dataSet, k)
    # 定义数组存储每个数据点的数据、与质点的距离、 簇编号
    dataCluter = np.zeros((dataSet.shape[0], 2))
    
    # 在更新质心后，需要重新对每个数据点分配簇
    # 因此，这里设置一个标志位，用来标志质心是否改变，是否需要重新分配；
    # 同时，当标志为False时，算法结束
    cluterSign = True
    while cluterSign:
        # 进来这一轮循环时，先将标志关闭，若后续簇的分配没有改变，则不再循环，算法结束
        # 若放最后，会将前边改变后的标志位关闭，结束算法
        cluterSign = False
        for i, d in enumerate(dataSet):
            # 定义每一个数据到质心的最小距离，将计算得到的距离和其比较，若小于将该点分配到该簇
            minDis = np.inf
            cluterNum = -1
            for j, c in enumerate(centre):
                dis = distance(d, c)
                if dis < minDis:
                    minDis = dis
                    cluterNum = j
            if dataCluter[i, 1] != cluterNum:
                # 当聚类的簇编号发生改变时，说明需要继续循环数据，因此设置 cluterSign=True
                cluterSign = True
                dataCluter[i,:] = minDis, cluterNum
        # 重新计算质心
        for n in range(k):
            # 找出每一簇中的数据
            dataC = dataSet[np.nonzero(dataCluter[:, 1] == n)[0]]
            # 重新计算质心, 对该簇内的数据按列求平均
            centre[n, :] = np.mean(dataC, axis=0)
    return dataCluter

def plotRes(data, clusterRes, k):
    """
    结果可视化
    :param data:样本集
    :param clusterRes:聚类结果
    :param k: 类个数
    :return:
    """
    nPoints = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(k):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')
    plt.show()

def test(k):
    '''
    算法验证
    :param k: 类个数
    '''
    path = 'D:/learn/AiLab/Kmeans/testSet.txt'
    dataSet = loadDataSet(path)
    dataCluter = kMmeans(dataSet, k)
    plotRes(dataSet, dataCluter[:, 1], k)
    
if __name__ == '__main__':
   test(3)
    
    
    
    
    
    