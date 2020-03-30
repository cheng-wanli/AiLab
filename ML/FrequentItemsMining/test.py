#!/usr/bin/env python3
# encoding: utf-8
"""
@time: 2020/3/26 11:06
@author: cheng
@file: test.py.py
"""
from apriori import Apriori


def aprioriTest(dataset):
    apriori = Apriori(dataset, 0.4)
    frequentItems, n = apriori.generateLk()
    associationRules = apriori.geneAssociationRules(0.01)
    return frequentItems, n, associationRules


if __name__ == '__ main __':
    dataSet = [['l1', 'l2', 'l5'], ['l2', 'l4'], ['l2', 'l3'],
               ['l1', 'l2', 'l4'], ['l1', 'l3'], ['l2', 'l3'],
               ['l1', 'l3'], ['l1', 'l2', 'l3', 'l5'], ['l1', 'l2', 'l3']]

    frequentItems, n, associationRules = aprioriTest(dataSet)
    print(frequentItems)
    print(n)
    print(associationRules)
