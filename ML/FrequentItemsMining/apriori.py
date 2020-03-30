#!/usr/bin/env python3
# encoding: utf-8
"""
@time: 2020/3/25 14:38
@author: cheng
@file: apriori.py
"""
from itertools import combinations

class Apriori():
    def __init__(self, dataset, minSup):
        """
        :param dataset: 数据集，形式为[[], [], [], ...]
        """
        self.minSup = minSup
        self.dataSet = dataset
        self.num = len(dataset)  # 数据集大小
        self.support = dict()  # 频繁项的支持度
        self.frequentItems = []  # 保存所有的频繁项

    def generateC1(self):
        """
        生成一次候选项
        :return: c1
        """
        c1 = {frozenset([i]) for item in self.dataSet for i in item}
        return c1

    def isFrequently(self, cksub, lksub1):
        flag = False
        # 生成所有的子集
        n = len(cksub) - 1
        cksubs = combinations(cksub, n)  # 返回一个元组
        for i in cksubs:
            if set(i) in lksub1:
                flag = True
            else:
                flag = False
                break
        return flag

    def generateCk(self, lksub1):
        """
        由k-1次频繁项生成K次候选项
        :param lksub1: k-1次频繁项
        :return: ck k次候选项
        """

        '''
        思路：要根据k-1次频繁项，遍历k-1次频繁项中的两两组合，找出符合（两个k-1次频繁项之间必须要有k-2个元素相同）条件的，拼接成K次候选项，
            1、遍历k-1次频繁项中的两两组合：
                使用两层循环，都针对k-1次频繁项集，但内层循环的起始索引比外层循环加1
            2、判断两个k-1次频繁项之间是否有k-2个元素是相同的：
                a）由于每个频繁项是frozenset集合存储的，集合是无序的，因此对其排序，排序后变为列表，比较前k-2项是否相同即可
                  （另一种思路：看两个频繁项的交集的长度是否等于k-2）
                b）若满足上述条件，将这两个频繁项求并集，得到一项候选项。
                c）再判断该条候选项的每一个子集是不是满足（如果某个项集是频繁的，那么它的所有子集也是频繁的）规则
                    即判断该候选项的子集是否在k-1次频繁项集中
                d）若在，加入候选集
        '''
        ck = set()
        lksub1List = list(lksub1)
        # 获得每个k-1次频繁项的长度
        k = len(lksub1List[0])
        for i in range(len(lksub1List)):
            for j in range(1, len(lksub1List)):
                # 排序
                lksub1Data1 = sorted(lksub1List[i])
                lksub1Data2 = sorted(lksub1List[j])
                # 判断两个k-1次频繁项之间是否有k-2个元素是相同的
                if lksub1Data1[:k - 1] == lksub1Data2[:k - 1]:
                    cksub = lksub1List[i] | lksub1List[j]
                    #  再判断该条候选项的每一个子集是不是满足（如果某个项集是频繁的，那么它的所有子集也是频繁的）规则
                    # 这里可能不存在该情况
                    if self.isFrequently(cksub, lksub1):
                        ck.add(cksub)
        return ck

    def generateLkByCk(self, ck):
        """
        根据k次候选项生成频繁项
        :param ck: k次候选项，集合
        :param minSup: 支持度阈值
        :return: lk 频繁项
        """

        # 计算每个候选项出现的次数
        itemCount = dict()
        for item in ck:
            for data in self.dataSet:
                if item.issubset(data):
                    if item not in itemCount:
                        itemCount[item] = 1
                    else:
                        itemCount[item] += 1

        lk = set()  # 频繁项,使用aet()存储，方便后续关联规则的计算
        for k, v in itemCount.items():
            if v / self.num >= self.minSup:
                lk.add(k)
                self.support[k] = v / self.num
        return lk

    def generateLk(self):
        # 生成一次候选项
        c1 = self.generateC1()
        # 由候选项生成频繁项
        l1 = self.generateLkByCk(c1)
        lksub1 = l1
        # 将频繁项和其对应的支持度保存为元组
        self.frequentItems.extend([(item, self.support[item]) for item in lksub1])

        # 最大频繁项的大小
        numMaxFrequent = 1

        # 要计算几次频繁项，循环几次即可
        while True:
            ck = self.generateCk(lksub1)
            lk = self.generateLkByCk(ck)
            lksub1 = lk
            if len(lksub1) != 0:
                numMaxFrequent += 1
                # self.frequentItems = [(item, self.support[item]) for item in lksub1]
                self.frequentItems.extend([(item, self.support[item]) for item in lksub1])
            else:
                break
        return self.frequentItems, numMaxFrequent

    def geneAssociationRules(self, minConf):
        """
        生成关联规则
        :param minConf: 最小置信度阈值
        :return: associationRules 关联规则
        """

        '''
        思想：要计算关联规则，即B->A的置信度是否大于最小的置信度，这里可以是多对多的关系
            conf(B->A) = support(A,B) / support(B)
            基于此，实际上只需计算每一项的支持度和其子集的支持的比即可。这里需要两个变量，
            一个变量a：当前项；另一个变量b：比其长度小的项
            由于频繁项集是按频繁项的长度从小到大排列的，变量a是变量b在频繁项集中的下一项。b中的项才可能是a的子集。
            每循环一个频繁项，就将该项加入b
        '''
        subSets = []       # 保存长度小于等于当前频繁项的项，然后遍历所有的项，判断是否是当前项的子集，然后计算其他项能推出该项的置信度
        associationRules = []
        for freqItem in self.frequentItems:
            # 由于freqItem格式为(frozenset({'l1'}), 0.6666666666666666)，只取频繁项
            freqItem = freqItem[0]
            for subItem in subSets:
                if subItem.issubset(freqItem):
                    conf = self.support[freqItem] / self.support[freqItem - subItem]
                    rule = (freqItem - subItem, subItem, conf)
                    if conf >= minConf and rule not in associationRules:
                        print(freqItem - subItem, '=>', subItem, 'conf:', conf)
                        associationRules.append(rule)
            subSets.append(freqItem)
        return associationRules
